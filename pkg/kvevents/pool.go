// Copyright 2025 The llm-d Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package kvevents

import (
	"context"
	"fmt"
	"hash/fnv"
	"strings"
	"sync"

	lru "github.com/hashicorp/golang-lru/v2"
	"k8s.io/client-go/util/workqueue"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/utils/logging"
)

const (
	defaultEventSourceDeviceTier = "GPU"
	defaultPodSelector           = "llm-d.ai/inference-serving=true"
)

// Config holds the configuration for the event processing pool.
type Config struct {
	// ZMQEndpoint is the ZMQ address to connect to (e.g., "tcp://indexer:5557").
	ZMQEndpoint string `json:"zmqEndpoint,omitempty"`
	// TopicFilter is the ZMQ subscription filter (e.g., "kv@").
	TopicFilter string `json:"topicFilter"`
	// Concurrency is the number of parallel workers to run.
	Concurrency int `json:"concurrency"`
	// EngineType selects the inference engine adapter ("vllm" or "sglang").
	// Default: "vllm".
	EngineType string `json:"engineType,omitempty"`
	// DiscoverPods enables the Kubernetes pod reconciler for automatic
	// per-pod subscriber management. When enabled, the reconciler watches
	// Kubernetes pods and creates/removes ZMQ subscribers dynamically.
	DiscoverPods bool `json:"discoverPods"`
	// PodDiscoveryConfig holds the configuration for pod discovery.
	// Only used when DiscoverPods is true.
	PodDiscoveryConfig *PodDiscoveryConfig `json:"podDiscoveryConfig,omitempty"`
}

// PodDiscoveryConfig holds configuration for the Kubernetes pod reconciler.
type PodDiscoveryConfig struct {
	// PodLabelSelector is a label selector string for filtering which pods to watch.
	// Example: "app=vllm" or "app=vllm,tier=gpu"
	PodLabelSelector string `json:"podLabelSelector"`
	// PodNamespace limits the reconciler to watch pods in a specific namespace.
	// If empty, watches all namespaces (requires appropriate RBAC).
	PodNamespace string `json:"podNamespace,omitempty"`
	// SocketPort is the port number where vLLM pods expose their ZMQ socket.
	// The reconciler will connect to tcp://<PodIP>:<SocketPort>
	// Default: 5557
	SocketPort int `json:"socketPort"`
}

// CheckpointAccumulator holds the data structure to track seen storage blocks per prefix.
type CheckpointAccumulator struct {
	counts *lru.Cache[kvblock.BlockHash, int]
	stride int
}

// NewCheckpointAccumulator instantiates a new checkpointAccumulator with a given stride and capacity.
func NewCheckpointAccumulator(stride, capacity int) (*CheckpointAccumulator, error) {
	if stride <= 0 {
		return nil, fmt.Errorf("stride must be greater than 0")
	}
	if capacity <= 0 {
		return nil, fmt.Errorf("capacity must be greater than 0")
	}

	counts, err := lru.New[kvblock.BlockHash, int](capacity)
	if err != nil {
		return nil, err
	}

	return &CheckpointAccumulator{counts: counts, stride: stride}, nil
}

// Accumulate adds blocks and returns the previous count, new count, and the
// checkpoint offsets crossed within this event slice.
func (a *CheckpointAccumulator) Accumulate(parentKey kvblock.BlockHash, numBlocks int) (int, int, []int) {
	current, _ := a.counts.Get(parentKey)
	if numBlocks <= 0 {
		return current, current, nil
	}

	newCount := current + numBlocks
	checkpointOffsets := make([]int, 0)

	for checkpoint := ((current / a.stride) + 1) * a.stride; checkpoint <= newCount; checkpoint += a.stride {
		offset := checkpoint - current - 1
		if offset >= 0 && offset < numBlocks {
			checkpointOffsets = append(checkpointOffsets, offset)
		}
	}

	return current, newCount, checkpointOffsets
}

// Update stores the new count keyed by the latest request key.
func (a *CheckpointAccumulator) Update(key kvblock.BlockHash, count int) {
	a.counts.Add(key, count)
}

// DefaultPodReconcilerConfig returns a default configuration for the pod reconciler.
func DefaultPodReconcilerConfig() *PodDiscoveryConfig {
	return &PodDiscoveryConfig{
		PodLabelSelector: defaultPodSelector,
		SocketPort:       5557,
	}
}

// DefaultConfig returns a default configuration for the event processing pool.
func DefaultConfig() *Config {
	return &Config{
		TopicFilter:        "kv@",
		Concurrency:        4,
		DiscoverPods:       true,
		PodDiscoveryConfig: DefaultPodReconcilerConfig(),
	}
}

// PoolOption is a functional option for configuring the Pool.
type PoolOption func(*Pool)

// gpuBatchData caches token data from GPU events so that storage events
// (which arrive without tokens) can resolve them via engine key correlation.
type gpuBatchData struct {
	tokens           []uint32
	modelName        string
	parentRequestKey kvblock.BlockHash
}

// WithStorageIndex returns a PoolOption that wires a pre-built storage index,
// checkpoint accumulator, and GPU token cache into the Pool.
func WithStorageIndex(
	storageIndex kvblock.Index,
	accumulator *CheckpointAccumulator,
	gpuTokenCache *lru.Cache[kvblock.BlockHash, *gpuBatchData],
) (PoolOption, error) {
	if storageIndex == nil {
		return nil, fmt.Errorf("storageIndex must not be nil")
	}
	if accumulator == nil {
		return nil, fmt.Errorf("storageAccumulator must not be nil")
	}
	if gpuTokenCache == nil {
		return nil, fmt.Errorf("gpuTokenCache must not be nil")
	}

	return func(p *Pool) {
		p.storageIndex = storageIndex
		p.storageAccumulator = accumulator
		p.gpuTokenCache = gpuTokenCache
	}, nil
}

// WithStorageConfig returns a PoolOption that creates a CheckpointAccumulator
// and GPU token cache, then wires them with the storage index into the Pool.
func WithStorageConfig(
	storageIndex kvblock.Index,
	checkpointStride, accumulatorCapacity, gpuTokenCacheCapacity int,
) (PoolOption, error) {
	accumulator, err := NewCheckpointAccumulator(checkpointStride, accumulatorCapacity)
	if err != nil {
		return nil, fmt.Errorf("failed to create storage accumulator: %w", err)
	}

	gpuTokenCache, err := lru.New[kvblock.BlockHash, *gpuBatchData](gpuTokenCacheCapacity)
	if err != nil {
		return nil, fmt.Errorf("failed to create GPU token cache: %w", err)
	}

	return WithStorageIndex(storageIndex, accumulator, gpuTokenCache)
}

// Pool is a sharded worker pool that processes events from ZMQ subscribers.
// It ensures that events for the same PodIdentifier are processed in order.
// Pool is stateless — all key mappings are delegated to the Index.
type Pool struct {
	queues             []workqueue.TypedRateLimitingInterface[*RawMessage]
	concurrency        int // can replace use with len(queues)
	index              kvblock.Index
	tokenProcessor     kvblock.TokenProcessor
	adapter            EngineAdapter
	wg                 sync.WaitGroup
	storageIndex       kvblock.Index
	storageAccumulator *CheckpointAccumulator
	gpuTokenCache      *lru.Cache[kvblock.BlockHash, *gpuBatchData]
}

// NewPool creates a Pool with a sharded worker setup.
// Subscribers are managed by SubscriberManager which is controlled by the pod
// reconciler.
func NewPool(cfg *Config, index kvblock.Index, tokenProcessor kvblock.TokenProcessor,
	adapter EngineAdapter, opts ...PoolOption,
) *Pool {
	if cfg == nil {
		cfg = DefaultConfig()
	}

	p := &Pool{
		queues:         make([]workqueue.TypedRateLimitingInterface[*RawMessage], cfg.Concurrency),
		concurrency:    cfg.Concurrency,
		index:          index,
		tokenProcessor: tokenProcessor,
		adapter:        adapter,
	}

	for i := 0; i < p.concurrency; i++ {
		p.queues[i] = workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[*RawMessage]())
	}

	for _, opt := range opts {
		if opt == nil {
			continue
		}
		opt(p)
	}

	return p
}

// Start begins the worker pool.
// It is non-blocking.
func (p *Pool) Start(ctx context.Context) {
	logger := log.FromContext(ctx)
	logger.Info("Starting sharded event processing pool", "workers", p.concurrency)

	p.wg.Add(p.concurrency)
	for i := 0; i < p.concurrency; i++ {
		// Each worker is given its own dedicated queue shard.
		go p.worker(ctx, i)
	}
}

// Shutdown gracefully stops the pool and its global subscriber if present.
func (p *Pool) Shutdown(ctx context.Context) {
	logger := log.FromContext(ctx)
	logger.Info("Shutting down event processing pool...")

	for _, queue := range p.queues {
		queue.ShutDown()
	}

	p.wg.Wait()
	logger.Info("event processing pool shut down.")
}

// AddTask is called by the subscriber to add a message to the processing queue.
// It hashes the sharding key to select a queue, ensuring messages for the
// same pod always go to the same worker (ordered queue).
func (p *Pool) AddTask(task *RawMessage) {
	key := p.adapter.ShardingKey(task)
	// Use an FNV-1a hash to deterministically select a queue.
	h := fnv.New32a()
	_, err := h.Write([]byte(key))
	if err != nil {
		return
	}

	//nolint:gosec // if concurrency overflows then the world is in trouble anyway
	queueIndex := h.Sum32() % uint32(p.concurrency)
	p.queues[queueIndex].Add(task)
}

// worker is the main processing loop for a single worker goroutine.
// It processes messages from its dedicated queue using the workqueue pattern.
func (p *Pool) worker(ctx context.Context, workerIndex int) {
	defer p.wg.Done()
	queue := p.queues[workerIndex]
	for {
		task, shutdown := queue.Get()
		if shutdown {
			return
		}

		// Use a nested func to ensure Done is always called.
		func(task *RawMessage) {
			defer queue.Done(task)
			p.processRawMessage(ctx, task)
			// Task succeeded, remove it from the queue.
			queue.Forget(task)
		}(task)

		// Check if context was cancelled after processing a task.
		select {
		case <-ctx.Done():
			return
		default:
		}
	}
}

// processRawMessage decodes the raw message payload using the adapter and processes the resulting event batch.
func (p *Pool) processRawMessage(ctx context.Context, msg *RawMessage) {
	logger := log.FromContext(ctx)

	podID, modelName, batch, err := p.adapter.ParseMessage(msg)
	if err != nil {
		logger.Error(err, "Failed to parse message")
		return
	}

	p.processEventBatch(ctx, &batch, podID, modelName)
}

// realignExtraFeatures converts per-engine-block extra features to per-canonical-block
// granularity so that len(result) matches the canonical chunk count expected by
// TokensToKVBlockKeys.
//
// For 1:many (engine BS > canonical BS): each engine block's features are replicated
// to all its constituent canonical sub-blocks.
// For many:1 (engine BS < canonical BS): features from multiple engine blocks are
// merged (union of MMHashes) into each canonical block.
//
// When all entries are nil (text-only prompts), this simply produces a nil-filled
// slice of the correct length.
func realignExtraFeatures(engineFeatures []*kvblock.BlockExtraFeatures, canonicalBlockCount int) []*kvblock.BlockExtraFeatures {
	engineBlockCount := len(engineFeatures)
	if engineBlockCount == canonicalBlockCount {
		return engineFeatures
	}

	canonical := make([]*kvblock.BlockExtraFeatures, canonicalBlockCount)

	if engineBlockCount < canonicalBlockCount {
		// 1:many -> replicate each engine feature to its canonical sub-blocks
		for i := range canonicalBlockCount {
			engineIdx := i * engineBlockCount / canonicalBlockCount
			canonical[i] = engineFeatures[engineIdx]
		}
	} else {
		// many:1 -> merge constituent engine features into each canonical block
		for i, ef := range engineFeatures {
			canonicalIdx := i * canonicalBlockCount / engineBlockCount
			if ef == nil {
				continue
			}
			if canonical[canonicalIdx] == nil {
				canonical[canonicalIdx] = &kvblock.BlockExtraFeatures{}
			}
			canonical[canonicalIdx].MMHashes = append(
				canonical[canonicalIdx].MMHashes, ef.MMHashes...)
		}
	}

	return canonical
}

// processEventBatch processes a batch of events using type switches.
func (p *Pool) processEventBatch(ctx context.Context, batch *EventBatch, podIdentifier, modelName string) {
	debugLogger := log.FromContext(ctx).V(logging.DEBUG)
	debugLogger.V(logging.TRACE).Info("Processing event batch",
		"podID", podIdentifier,
		"modelName", modelName,
		"eventCount", len(batch.Events))

	// Process each event in the batch
	for _, genericEvent := range batch.Events {
		switch ev := genericEvent.(type) {
		case *BlockStoredEvent:
			// Default to gpu.
			deviceTier := defaultEventSourceDeviceTier
			if ev.DeviceTier != "" {
				deviceTier = strings.ToLower(ev.DeviceTier)
			}

			// Use LoRA name as model identifier if available, otherwise fall back to base model name.
			effectiveModelName := modelName
			if ev.LoraName != nil && *ev.LoraName != "" {
				effectiveModelName = *ev.LoraName
			}

			// Create PodEntry for this specific event's device tier
			podEntries := []kvblock.PodEntry{{PodIdentifier: podIdentifier, DeviceTier: deviceTier}}

			engineKeys := make([]kvblock.BlockHash, len(ev.BlockHashes))
			for i, hash := range ev.BlockHashes {
				engineKeys[i] = kvblock.BlockHash(hash)
			}

			storageTier := isStorageTier(deviceTier) && p.storageIndex != nil

			parentRequestKey := kvblock.EmptyBlockHash
			if !storageTier && ev.ParentHash != 0 {
				parentEngineKey := kvblock.BlockHash(ev.ParentHash)
				// Try GPU/CPU index first, then storage index
				key, err := p.index.GetRequestKey(ctx, parentEngineKey)
				if err != nil && p.storageIndex != nil {
					key, err = p.storageIndex.GetRequestKey(ctx, parentEngineKey)
				}
				if err != nil {
					debugLogger.Error(err, "Failed to get request key for parent block",
						"parentEngineKey", parentEngineKey, "effectiveModelName", effectiveModelName)
					continue
				}
				parentRequestKey = key
			}

			if storageTier {
				// Storage path: resolve tokens from GPU cache (storage events don't carry tokens).
				data := p.lookupStorageTokenData(engineKeys)
				if data == nil {
					debugLogger.Info("No cached GPU data for storage event, skipping",
						"podIdentifier", podIdentifier, "firstEngineKey", engineKeys[0])
					continue
				}

				// Storage events are text-only, no extraFeatures.
				requestKeys, err := p.tokenProcessor.TokensToKVBlockKeys(
					data.parentRequestKey, data.tokens, data.modelName, nil)
				if err != nil {
					debugLogger.Error(err, "Failed to generate request keys for storage event",
						"podIdentifier", podIdentifier, "effectiveModelName", data.modelName)
					continue
				}

				if len(requestKeys) > 0 {
					lastKey := requestKeys[len(requestKeys)-1]
					_, newCount, checkpointOffsets := p.storageAccumulator.Accumulate(parentRequestKey, len(requestKeys))
					p.storageAccumulator.Update(lastKey, newCount)

					storageEntries := []kvblock.PodEntry{{PodIdentifier: "shared-storage", DeviceTier: "storage"}}
					for _, offset := range checkpointOffsets {
						if offset >= len(requestKeys) || offset >= len(engineKeys) {
							debugLogger.Error(nil, "Storage checkpoint offset out of bounds",
								"podIdentifier", podIdentifier, "offset", offset,
								"requestKeyCount", len(requestKeys), "engineKeyCount", len(engineKeys))
							continue
						}
						cpEngineKeys := []kvblock.BlockHash{engineKeys[offset]}
						cpRequestKeys := []kvblock.BlockHash{requestKeys[offset]}
						if err := p.storageIndex.Add(ctx, cpEngineKeys, cpRequestKeys, storageEntries); err != nil {
							debugLogger.Error(err, "Failed to add storage checkpoint to index",
								"podIdentifier", podIdentifier,
								"engineKey", engineKeys[offset],
								"requestKey", requestKeys[offset])
						}
					}
				}
			} else {
				// Existing GPU/CPU path
				var extraFeatures []*kvblock.BlockExtraFeatures
				if ev.ExtraKeys != nil {
					var err error
					extraFeatures, err = kvblock.ParseRawExtraKeys(ev.ExtraKeys)
					if err != nil {
						debugLogger.Error(err, "Failed to parse extra keys",
							"podIdentifier", podIdentifier)
						continue
					}
				}

				// Realign extraFeatures from engine-block granularity to canonical-block
				// granularity. ParseRawExtraKeys returns one entry per engine block, but
				// TokensToKVBlockKeys expects one entry per canonical block.
				if extraFeatures != nil {
					canonicalBlockCount := len(ev.Tokens) / p.tokenProcessor.BlockSize()
					if len(extraFeatures) != canonicalBlockCount {
						extraFeatures = realignExtraFeatures(extraFeatures, canonicalBlockCount)
					}
				}

				traceLogger := log.FromContext(ctx).V(logging.TRACE)
				if traceLogger.Enabled() {
					nonNil := 0
					for _, ef := range extraFeatures {
						if ef != nil {
							nonNil++
						}
					}
					traceLogger.Info("BlockStored extra_features",
						"podIdentifier", podIdentifier,
						"hasExtraKeys", ev.ExtraKeys != nil,
						"parsedBlockCount", len(extraFeatures),
						"nonNilBlocks", nonNil,
						"numTokens", len(ev.Tokens),
						"numEngineKeys", len(ev.BlockHashes))
					for bIdx, ef := range extraFeatures {
						if ef != nil {
							traceLogger.Info("BlockStored block extra",
								"podIdentifier", podIdentifier,
								"blockIdx", bIdx,
								"mmHashes", fmt.Sprintf("%+v", ef.MMHashes))
						}
					}
				}

				// Compute request keys at canonical block size (= BlockSize)
				requestKeys, err := p.tokenProcessor.TokensToKVBlockKeys(
					parentRequestKey, ev.Tokens, effectiveModelName, extraFeatures)
				if err != nil {
					debugLogger.Error(err, "Failed to generate request keys",
						"podIdentifier", podIdentifier, "effectiveModelName", effectiveModelName)
					continue
				}

				if len(requestKeys) == 0 {
					debugLogger.Info("no request keys produced, skipping",
						"podIdentifier", podIdentifier, "tokenCount", len(ev.Tokens),
						"blockSize", p.tokenProcessor.BlockSize())
					continue
				}

				// Index.Add infers the engine->request mapping from the ratio of
				// len(engineKeys) to len(requestKeys) (1:1, many:1, or 1:many).
				if err := p.index.Add(ctx, engineKeys, requestKeys, podEntries); err != nil {
					debugLogger.Error(err, "Failed to add event to index",
						"podIdentifier", podIdentifier, "event", ev)
					continue
				}

				// Cache token data for later storage event correlation.
				if p.gpuTokenCache != nil && len(engineKeys) > 0 {
					p.gpuTokenCache.Add(engineKeys[0], &gpuBatchData{
						tokens:           ev.Tokens,
						modelName:        effectiveModelName,
						parentRequestKey: parentRequestKey,
					})
				}
			}

		case *BlockRemovedEvent:
			// Default to gpu.
			deviceTier := defaultEventSourceDeviceTier
			if ev.DeviceTier != "" {
				deviceTier = strings.ToLower(ev.DeviceTier)
			}

			if isStorageTier(deviceTier) && p.storageIndex != nil {
				storageEntries := []kvblock.PodEntry{{PodIdentifier: "shared-storage", DeviceTier: "storage"}}
				for _, hash := range ev.BlockHashes {
					engineKey := kvblock.BlockHash(hash)
					if err := p.storageIndex.Evict(ctx, engineKey, kvblock.EngineKey, storageEntries); err != nil {
						debugLogger.Error(err, "Failed to remove event from storage index",
							"podIdentifier", podIdentifier, "engineKey", engineKey, "event", ev)
						continue
					}
				}
			} else {

				// Create PodEntry for this specific event's device tier
				podEntries := []kvblock.PodEntry{{PodIdentifier: podIdentifier, DeviceTier: deviceTier}}

				// Iterate over the hashes and evict each key.
				// The Index handles engine->request key resolution internally for both
				// 1:1 (legacy) and 1:many (canonical) mappings.
				for _, hash := range ev.BlockHashes {
					engineKey := kvblock.BlockHash(hash)
					if err := p.index.Evict(ctx, engineKey, kvblock.EngineKey, podEntries); err != nil {
						debugLogger.Error(err, "Failed to evict engine key from index",
							"podIdentifier", podIdentifier, "engineKey", engineKey)
						continue
					}
				}
			}

		case *AllBlocksClearedEvent:
			debugLogger.Info("All blocks cleared event received",
				"podIdentifier", podIdentifier,
				"deviceTier", ev.DeviceTier,
				"modelName", modelName)

		default:
			debugLogger.Info("Unknown event", "podIdentifier", podIdentifier, "event", genericEvent)
		}
	}
}

// lookupStorageTokenData resolves token data for a storage event by looking up
// the first engine key in the GPU token cache. Storage events arrive without
// tokens; the tokens must have been cached from an earlier GPU event.
func (p *Pool) lookupStorageTokenData(engineKeys []kvblock.BlockHash) *gpuBatchData {
	if p.gpuTokenCache == nil || len(engineKeys) == 0 {
		return nil
	}
	data, ok := p.gpuTokenCache.Get(engineKeys[0])
	if !ok {
		return nil
	}
	return data
}

func isStorageTier(tier string) bool {
	return tier == "shared_storage" || tier == "local_storage"
}

func isGPUTier(tier string) bool {
	return strings.EqualFold(tier, "gpu")
}
