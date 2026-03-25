/*
Copyright 2025 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package kvblock

import (
	"context"
	"encoding/binary"
	"fmt"
	"sync"

	lru "github.com/hashicorp/golang-lru/v2"
	"github.com/llm-d/llm-d-kv-cache/pkg/utils/logging"
	cuckoo "github.com/seiflotfy/cuckoofilter"
	"k8s.io/apimachinery/pkg/util/sets"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	defaultFilterCapacity   = 1e7
	defaultEngineKeyMapSize = 1e7
)

// CuckooStorageIndexConfig holds the configuration for the CuckooStorageIndex.
type CuckooStorageIndexConfig struct {
	// Capacity is the max number of entries the Cuckoo filter can hold.
	// // At ~95% occupancy, insertions may fail. Size with headroom.
	FilterCapacity uint `json:"filterCapacity"`
	// EngineKeyMapSize is the size of the engine->request LRU.
	EngineKeyMapSize int `json:"engineKeyMapSize"`
	// DefaultEntries are the entries returned for every lookup hit.
	// For shared storage: [{PodIdentifier: "shared-storage", DeviceTier: "storage"}]
	// For local NVMe: [{PodIdentifier: "<pod-id>", DeviceTier: "storage"}]
	DefaultEntries []PodEntry `json:"defaultEntries"`
}

// DefaultCuckooStorageIndexConfig returns a default configuration for the CuckooStorageIndex.
func DefaultCuckooStorageIndexConfig() *CuckooStorageIndexConfig {
	return &CuckooStorageIndexConfig{
		FilterCapacity:   defaultFilterCapacity,
		EngineKeyMapSize: defaultEngineKeyMapSize,
		DefaultEntries: []PodEntry{
			{PodIdentifier: "shared-storage", DeviceTier: "storage"},
		},
	}
}

// NewCuckooStorageIndex creates a new CuckooStorageIndex instance.
func NewCuckooStorageIndex(cfg *CuckooStorageIndexConfig) (*CuckooStorageIndex, error) {
	if cfg == nil {
		cfg = DefaultCuckooStorageIndexConfig()
	}
	if len(cfg.DefaultEntries) == 0 {
		return nil, fmt.Errorf("defaultEntries must not be empty")

	}

	engineMap, err := lru.New[BlockHash, BlockHash](cfg.EngineKeyMapSize)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize engine key map: %w", err)
	}

	return &CuckooStorageIndex{
		filter:              cuckoo.NewFilter(cfg.FilterCapacity),
		engineToRequestKeys: engineMap,
		defaultEntries:      cfg.DefaultEntries,
	}, nil
}

// CuckooStorageIndex implements the Index interface using a Cuckoo filter
// for compact set-membership queries with native deletion.
type CuckooStorageIndex struct {
	filter              *cuckoo.Filter
	engineToRequestKeys *lru.Cache[BlockHash, BlockHash]
	defaultEntries      []PodEntry
	mu                  sync.RWMutex
}

var _ Index = &CuckooStorageIndex{}

// Add inserts request keys into the Cuckoo filter and stores engine→request key mappings.
// The entries parameter is accepted for interface compliance but not stored per-key;
// Lookup returns the defaultEntries configured at construction.
func (c *CuckooStorageIndex) Add(ctx context.Context, engineKeys, requestKeys []BlockHash, entries []PodEntry) error {
	if len(requestKeys) == 0 || len(entries) == 0 {
		return fmt.Errorf("no keys or entries provided for adding to index")
	}
	if engineKeys != nil && len(engineKeys) != len(requestKeys) {
		return fmt.Errorf("mismatch between engine keys and request keys length")
	}

	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("kvblock.CuckooStorageIndex.Add")

	c.mu.Lock()
	defer c.mu.Unlock()

	for i, requestKey := range requestKeys {
		if engineKeys != nil {
			c.engineToRequestKeys.Add(engineKeys[i], requestKey)
		}

		if ok := c.filter.Insert(blockHashToBytes(requestKey)); !ok {
			return fmt.Errorf("cuckoo filter insertion failed for key %s (filter may be full)", requestKey.String())
		}

		traceLogger.Info("added key to storage index", "requestKey", requestKey, "entries", entries)
	}

	return nil
}

// GetRequestKey returns the requestKey associated with the given engineKey.
func (c *CuckooStorageIndex) GetRequestKey(_ context.Context, engineKey BlockHash) (BlockHash, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	requestKey, found := c.engineToRequestKeys.Get(engineKey)
	if !found {
		return EmptyBlockHash, fmt.Errorf("engine key not found: %s", engineKey.String())
	}
	return requestKey, nil
}

// Lookup checks each request against the Cuckoo filter.
// It returns all the keys, since no filtering is either needed nor make sense for shared storage.
func (c *CuckooStorageIndex) Lookup(ctx context.Context, requestKeys []BlockHash, _ sets.Set[string]) (map[BlockHash][]PodEntry, error) {
	if len(requestKeys) == 0 {
		return nil, fmt.Errorf("no keys provided for lookup")
	}

	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("kvblock.CuckooStorageIndex.Lookup")

	c.mu.RLock()
	defer c.mu.RUnlock()

	podsPerKey := make(map[BlockHash][]PodEntry)
	highestHitIdx := 0

	for idx, requestKey := range requestKeys {
		if c.filter.Lookup(blockHashToBytes(requestKey)) {
			highestHitIdx = idx
			podsPerKey[requestKey] = c.defaultEntries
		} else {
			traceLogger.Info("key not found in storage index", "key", requestKey)
		}
	}

	traceLogger.Info("lookup completed", "highest-hit-index", highestHitIdx, "hits", len(podsPerKey))

	return podsPerKey, nil
}

// Evict removes a key from the Cuckoo filter.
// For EngineKey type, resolves to requestKey via de internal LRU first
// Note: Cuckoo filter removes the entire key, no partial eviction is not supported nor needed.
func (c *CuckooStorageIndex) Evict(ctx context.Context, key BlockHash, keyType KeyType, _ []PodEntry) error {
	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("kvblock.CuckooStorageIndex.Evict")

	c.mu.Lock()
	defer c.mu.Unlock()

	var requestKey BlockHash
	switch keyType {
	case EngineKey:
		rk, found := c.engineToRequestKeys.Get(key)
		if !found {
			traceLogger.Info("engineKey not found in mapping, nothing to evict", "engineKey", key)
			return nil
		}
		requestKey = rk
		c.engineToRequestKeys.Remove(key)
	case RequestKey:
		requestKey = key
	default:
		return fmt.Errorf("unknown key type: %d", keyType)
	}

	c.filter.Delete(blockHashToBytes(requestKey))
	traceLogger.Info("evicted key from storage index", "requestKey", requestKey, "key", key, "keyType", keyType)
	return nil
}

// Helper to convert blockHash to []byte
func blockHashToBytes(h BlockHash) []byte {
	b := make([]byte, 8)
	binary.BigEndian.PutUint64(b, uint64(h))
	return b
}
