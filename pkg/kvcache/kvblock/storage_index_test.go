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

package kvblock_test

import (
	"fmt"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/sets"

	. "github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/utils/logging"
)

// defaultStorageEntries is the fixed set of PodEntries used by the storage index in tests.
var defaultStorageEntries = []PodEntry{{PodIdentifier: "shared-storage", DeviceTier: "storage"}}

// createCuckooStorageIndexForTesting creates a CuckooStorageIndex with default config.
func createCuckooStorageIndexForTesting(t *testing.T) Index {
	t.Helper()
	cfg := DefaultCuckooStorageIndexConfig()
	idx, err := NewCuckooStorageIndex(cfg)
	require.NoError(t, err)
	return idx
}

func TestCuckooStorageIndex_NewValidation(t *testing.T) {
	t.Run("nil config uses defaults", func(t *testing.T) {
		idx, err := NewCuckooStorageIndex(nil)
		require.NoError(t, err)
		assert.NotNil(t, idx)
	})

	t.Run("empty defaultEntries returns error", func(t *testing.T) {
		cfg := &CuckooStorageIndexConfig{
			FilterCapacity:   100000,
			EngineKeyMapSize: 100000,
			DefaultEntries:   []PodEntry{},
		}
		idx, err := NewCuckooStorageIndex(cfg)
		assert.Error(t, err)
		assert.Nil(t, idx)
		assert.Contains(t, err.Error(), "defaultEntries")
	})
}

func TestCuckooStorageIndex_BasicAddAndLookup(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	idx := createCuckooStorageIndexForTesting(t)

	engineKey := BlockHash(55269488)
	requestKey := BlockHash(10633516)

	err := idx.Add(ctx, []BlockHash{engineKey}, []BlockHash{requestKey}, defaultStorageEntries)
	require.NoError(t, err)

	podsPerKey, err := idx.Lookup(ctx, []BlockHash{requestKey}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Len(t, podsPerKey, 1)
	assert.Contains(t, podsPerKey, requestKey)
	assert.ElementsMatch(t, podsPerKey[requestKey], defaultStorageEntries)
}

func TestCuckooStorageIndex_LookupMiss(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	idx := createCuckooStorageIndexForTesting(t)

	missingKey := BlockHash(99999)
	podsPerKey, err := idx.Lookup(ctx, []BlockHash{missingKey}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Empty(t, podsPerKey)
}

func TestCuckooStorageIndex_LookupIgnoresPodIdentifierSetForSharedStorage(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	idx := createCuckooStorageIndexForTesting(t)

	requestKey := BlockHash(123456)
	err := idx.Add(ctx, nil, []BlockHash{requestKey}, defaultStorageEntries)
	require.NoError(t, err)

	podsPerKey, err := idx.Lookup(ctx, []BlockHash{requestKey}, sets.New[string]("unrelated-pod"))
	require.NoError(t, err)
	require.Contains(t, podsPerKey, requestKey)
	assert.ElementsMatch(t, defaultStorageEntries, podsPerKey[requestKey])
}

func TestCuckooStorageIndex_LookupEmptyRequestKeys(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	idx := createCuckooStorageIndexForTesting(t)

	_, err := idx.Lookup(ctx, []BlockHash{}, sets.Set[string]{})
	assert.Error(t, err)
}

func TestCuckooStorageIndex_AddValidation(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	idx := createCuckooStorageIndexForTesting(t)

	t.Run("empty requestKeys returns error", func(t *testing.T) {
		err := idx.Add(ctx, nil, []BlockHash{}, defaultStorageEntries)
		assert.Error(t, err)
	})

	t.Run("empty entries returns error", func(t *testing.T) {
		err := idx.Add(ctx, nil, []BlockHash{BlockHash(1)}, []PodEntry{})
		assert.Error(t, err)
	})

	t.Run("mismatched engineKeys and requestKeys returns error", func(t *testing.T) {
		err := idx.Add(ctx,
			[]BlockHash{BlockHash(1), BlockHash(2)},
			[]BlockHash{BlockHash(3)},
			defaultStorageEntries,
		)
		assert.Error(t, err)
	})
}

func TestCuckooStorageIndex_MultipleKeysAddAndLookup(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	idx := createCuckooStorageIndexForTesting(t)

	engineKeys := []BlockHash{BlockHash(1001), BlockHash(1002), BlockHash(1003)}
	requestKeys := []BlockHash{BlockHash(2001), BlockHash(2002), BlockHash(2003)}

	err := idx.Add(ctx, engineKeys, requestKeys, defaultStorageEntries)
	require.NoError(t, err)

	// All three should be found
	podsPerKey, err := idx.Lookup(ctx, requestKeys, sets.Set[string]{})
	require.NoError(t, err)
	assert.Len(t, podsPerKey, 3)
	for _, rk := range requestKeys {
		assert.Contains(t, podsPerKey, rk)
		assert.ElementsMatch(t, podsPerKey[rk], defaultStorageEntries)
	}
}

func TestCuckooStorageIndex_EvictByEngineKey(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	idx := createCuckooStorageIndexForTesting(t)

	engineKey := BlockHash(11111)
	requestKey := BlockHash(22222)

	// Add and verify present
	err := idx.Add(ctx, []BlockHash{engineKey}, []BlockHash{requestKey}, defaultStorageEntries)
	require.NoError(t, err)

	podsPerKey, err := idx.Lookup(ctx, []BlockHash{requestKey}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Len(t, podsPerKey, 1)

	// Evict via engine key
	err = idx.Evict(ctx, engineKey, EngineKey, defaultStorageEntries)
	require.NoError(t, err)

	// Verify removed
	podsPerKey, err = idx.Lookup(ctx, []BlockHash{requestKey}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Empty(t, podsPerKey)
}

func TestCuckooStorageIndex_EvictByRequestKey(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	idx := createCuckooStorageIndexForTesting(t)

	requestKey := BlockHash(33333)

	// Add with nil engineKeys
	err := idx.Add(ctx, nil, []BlockHash{requestKey}, defaultStorageEntries)
	require.NoError(t, err)

	// Evict by request key directly
	err = idx.Evict(ctx, requestKey, RequestKey, defaultStorageEntries)
	require.NoError(t, err)

	podsPerKey, err := idx.Lookup(ctx, []BlockHash{requestKey}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Empty(t, podsPerKey)
}

func TestCuckooStorageIndex_EvictUnknownKeyIsNoop(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	idx := createCuckooStorageIndexForTesting(t)

	// Evict engine key that was never added — should not error
	err := idx.Evict(ctx, BlockHash(99999), EngineKey, defaultStorageEntries)
	require.NoError(t, err)

	// Evict request key that was never added — should not error
	err = idx.Evict(ctx, BlockHash(88888), RequestKey, defaultStorageEntries)
	require.NoError(t, err)
}

func TestCuckooStorageIndex_EvictRemovesEntireKey(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	idx := createCuckooStorageIndexForTesting(t)

	engineKey := BlockHash(44444)
	requestKey := BlockHash(55555)

	err := idx.Add(ctx, []BlockHash{engineKey}, []BlockHash{requestKey}, defaultStorageEntries)
	require.NoError(t, err)

	// Evict with a different PodEntry than what was added — storage eviction
	// removes the entire key regardless of entries parameter
	differentEntries := []PodEntry{{PodIdentifier: "other-pod", DeviceTier: "gpu"}}
	err = idx.Evict(ctx, engineKey, EngineKey, differentEntries)
	require.NoError(t, err)

	// Key should be fully removed
	podsPerKey, err := idx.Lookup(ctx, []BlockHash{requestKey}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Empty(t, podsPerKey)
}

func TestCuckooStorageIndex_GetRequestKey(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	idx := createCuckooStorageIndexForTesting(t)

	engineKey := BlockHash(44444)
	requestKey := BlockHash(55555)

	err := idx.Add(ctx, []BlockHash{engineKey}, []BlockHash{requestKey}, defaultStorageEntries)
	require.NoError(t, err)

	got, err := idx.GetRequestKey(ctx, engineKey)
	require.NoError(t, err)
	assert.Equal(t, requestKey, got)

	// Unknown engine key returns error
	_, err = idx.GetRequestKey(ctx, BlockHash(99999))
	assert.Error(t, err)
}

func TestCuckooStorageIndex_AddWithNilEngineKeys(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	idx := createCuckooStorageIndexForTesting(t)

	requestKey := BlockHash(66666)

	// Add with nil engineKeys should work
	err := idx.Add(ctx, nil, []BlockHash{requestKey}, defaultStorageEntries)
	require.NoError(t, err)

	// Lookup should find the entry
	podsPerKey, err := idx.Lookup(ctx, []BlockHash{requestKey}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Len(t, podsPerKey[requestKey], 1)
	assert.Contains(t, podsPerKey[requestKey], defaultStorageEntries[0])

	// GetRequestKey should NOT find a mapping (no engineKey was stored)
	_, err = idx.GetRequestKey(ctx, requestKey)
	assert.Error(t, err)
}

func TestCuckooStorageIndex_ConsecutivePrefixWalk(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	idx := createCuckooStorageIndexForTesting(t)

	// Simulate checkpoint keys at stride boundaries.
	// In the real flow, these are every stride-th storage-resolution key.
	checkpointKeys := []BlockHash{BlockHash(3001), BlockHash(3002), BlockHash(3003)}

	// Insert first two checkpoints but NOT the third
	for _, key := range checkpointKeys[:2] {
		err := idx.Add(ctx, nil, []BlockHash{key}, defaultStorageEntries)
		require.NoError(t, err)
	}

	// Lookup all three — should hit first two, miss third
	podsPerKey, err := idx.Lookup(ctx, checkpointKeys, sets.Set[string]{})
	require.NoError(t, err)

	assert.Contains(t, podsPerKey, checkpointKeys[0])
	assert.Contains(t, podsPerKey, checkpointKeys[1])
	assert.NotContains(t, podsPerKey, checkpointKeys[2])

	// The caller (scorer/indexer) interprets this as: consecutive prefix
	// up to checkpoint 1, gap at checkpoint 2.
}

func TestCuckooStorageIndex_EvictCleansUpEngineKeyMapping(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	idx := createCuckooStorageIndexForTesting(t)

	engineKey := BlockHash(77777)
	requestKey := BlockHash(88888)

	err := idx.Add(ctx, []BlockHash{engineKey}, []BlockHash{requestKey}, defaultStorageEntries)
	require.NoError(t, err)

	// Verify engine→request mapping exists
	got, err := idx.GetRequestKey(ctx, engineKey)
	require.NoError(t, err)
	assert.Equal(t, requestKey, got)

	// Evict via engine key
	err = idx.Evict(ctx, engineKey, EngineKey, defaultStorageEntries)
	require.NoError(t, err)

	// Engine→request mapping should be cleaned up
	_, err = idx.GetRequestKey(ctx, engineKey)
	assert.Error(t, err)
}

func TestCuckooStorageIndex_ReAddAfterEvict(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	idx := createCuckooStorageIndexForTesting(t)

	engineKey := BlockHash(10001)
	requestKey := BlockHash(20001)

	// Add, evict, then re-add
	err := idx.Add(ctx, []BlockHash{engineKey}, []BlockHash{requestKey}, defaultStorageEntries)
	require.NoError(t, err)

	err = idx.Evict(ctx, engineKey, EngineKey, defaultStorageEntries)
	require.NoError(t, err)

	// Verify gone
	podsPerKey, err := idx.Lookup(ctx, []BlockHash{requestKey}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Empty(t, podsPerKey)

	// Re-add with a new engine key
	newEngineKey := BlockHash(10002)
	err = idx.Add(ctx, []BlockHash{newEngineKey}, []BlockHash{requestKey}, defaultStorageEntries)
	require.NoError(t, err)

	// Should be back
	podsPerKey, err = idx.Lookup(ctx, []BlockHash{requestKey}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Len(t, podsPerKey, 1)
	assert.Contains(t, podsPerKey, requestKey)
}

func TestCuckooStorageIndex_ConcurrentAccess(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	idx := createCuckooStorageIndexForTesting(t)

	var wg sync.WaitGroup
	errChan := make(chan error, 1000)

	for goroutineID := range 100 {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for op := range 10 {
				key := BlockHash(uint64(id*1000 + op))
				switch op % 3 {
				case 0: // Add
					if err := idx.Add(ctx, nil, []BlockHash{key}, defaultStorageEntries); err != nil {
						errChan <- err
					}
				case 1: // Lookup
					if _, err := idx.Lookup(ctx, []BlockHash{key}, sets.Set[string]{}); err != nil {
						errChan <- err
					}
				case 2: // Evict
					if err := idx.Evict(ctx, key, RequestKey, defaultStorageEntries); err != nil {
						errChan <- err
					}
				}
			}
		}(goroutineID)
	}

	wg.Wait()
	close(errChan)

	for err := range errChan {
		require.NoError(t, err)
	}

	// Verify index still works after concurrent hammering
	_, err := idx.Lookup(ctx, []BlockHash{BlockHash(1)}, sets.Set[string]{})
	require.NoError(t, err)
}

func TestCuckooStorageIndex_FalsePositiveRate(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())

	cfg := &CuckooStorageIndexConfig{
		FilterCapacity:   100000,
		EngineKeyMapSize: 100000,
		DefaultEntries:   defaultStorageEntries,
	}
	idx, err := NewCuckooStorageIndex(cfg)
	require.NoError(t, err)

	// Insert 10K known keys
	numInserted := 10000
	for i := range numInserted {
		key := BlockHash(uint64(i))
		err := idx.Add(ctx, nil, []BlockHash{key}, defaultStorageEntries)
		require.NoError(t, err)
	}

	// Query 10K keys that were NEVER inserted (offset by numInserted)
	numQueries := 10000
	falsePositives := 0
	for i := numInserted; i < numInserted+numQueries; i++ {
		key := BlockHash(uint64(i))
		podsPerKey, err := idx.Lookup(ctx, []BlockHash{key}, sets.Set[string]{})
		require.NoError(t, err)
		if len(podsPerKey) > 0 {
			falsePositives++
		}
	}

	fpRate := float64(falsePositives) / float64(numQueries)
	t.Logf("False positive rate: %.4f%% (%d/%d)", fpRate*100, falsePositives, numQueries)

	// Cuckoo filter at 10% occupancy should have False Positives Rate well below 2%
	assert.Less(t, fpRate, 0.02, "False positive rate should be below 2%%")
}

func TestCuckooStorageIndex_LookupReturnsDefaultEntries(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())

	// Configure with multiple default entries (e.g., local NVMe per-pod scenario)
	multiEntries := []PodEntry{
		{PodIdentifier: "pod-a", DeviceTier: "storage"},
		{PodIdentifier: "pod-b", DeviceTier: "storage"},
	}
	cfg := &CuckooStorageIndexConfig{
		FilterCapacity:   100000,
		EngineKeyMapSize: 100000,
		DefaultEntries:   multiEntries,
	}
	idx, err := NewCuckooStorageIndex(cfg)
	require.NoError(t, err)

	requestKey := BlockHash(12345)
	err = idx.Add(ctx, nil, []BlockHash{requestKey}, multiEntries)
	require.NoError(t, err)

	// Lookup without filter returns all default entries
	podsPerKey, err := idx.Lookup(ctx, []BlockHash{requestKey}, sets.Set[string]{})
	require.NoError(t, err)
	assert.ElementsMatch(t, podsPerKey[requestKey], multiEntries)
}

func TestCuckooStorageIndex_EvictInvalidKeyType(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	idx := createCuckooStorageIndexForTesting(t)

	err := idx.Evict(ctx, BlockHash(1), KeyType(99), defaultStorageEntries)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "unknown key type")
}

func TestCuckooStorageIndex_LargeScale(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())

	// Simulate realistic checkpoint count: 100K entries
	cfg := &CuckooStorageIndexConfig{
		FilterCapacity:   1_000_000,
		EngineKeyMapSize: 1_000_000,
		DefaultEntries:   defaultStorageEntries,
	}
	idx, err := NewCuckooStorageIndex(cfg)
	require.NoError(t, err)

	numCheckpoints := 100000
	keys := make([]BlockHash, numCheckpoints)
	for i := range numCheckpoints {
		keys[i] = BlockHash(uint64(i * 7919)) // spread hashes
		err := idx.Add(ctx, nil, []BlockHash{keys[i]}, defaultStorageEntries)
		require.NoError(t, err)
	}

	// All should be found
	podsPerKey, err := idx.Lookup(ctx, keys, sets.Set[string]{})
	require.NoError(t, err)
	assert.Len(t, podsPerKey, numCheckpoints,
		fmt.Sprintf("Expected all %d checkpoint keys to be found", numCheckpoints))
}
