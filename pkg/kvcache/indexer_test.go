/*
Copyright 2026 The llm-d Authors.

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

package kvcache_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization/types"
	"github.com/llm-d/llm-d-kv-cache/pkg/utils/logging"
)

// --- mock implementations ---------------------------------------------------

// mockTokenProcessor implements kvblock.TokenProcessor for testing.
// It records the tokens it receives so tests can assert on them.
type mockTokenProcessor struct {
	blockKeys      []kvblock.BlockHash
	receivedTokens []uint32
}

func (m *mockTokenProcessor) TokensToKVBlockKeys(
	_ kvblock.BlockHash, tokens []uint32, _ string, _ []*kvblock.BlockExtraFeatures,
) ([]kvblock.BlockHash, error) {
	m.receivedTokens = tokens
	return m.blockKeys, nil
}

func (m *mockTokenProcessor) BlockSize() int {
	return 16
}

// mockTokenizersPool implements kvcache.TokenizersPool for testing.
type mockTokenizersPool struct {
	tokens []uint32
}

func (m *mockTokenizersPool) Tokenize(_ *types.RenderChatRequest, _ string) ([]uint32, *tokenization.MultiModalFeatures) {
	return m.tokens, nil
}

func (m *mockTokenizersPool) Run(_ context.Context) {}

func (m *mockTokenizersPool) SetTokenizer(_ tokenization.Tokenizer, _ string) {}

// --- helpers ----------------------------------------------------------------

const (
	testModel = "test-model"
	testPodA  = "pod-a"
	testPodB  = "pod-b"
)

func u64ToBlockKeys(keys []uint64) []kvblock.BlockHash {
	out := make([]kvblock.BlockHash, len(keys))
	for i, k := range keys {
		out[i] = kvblock.BlockHash(k)
	}
	return out
}

// newTestIndexer creates an Indexer backed by an in-memory index, a mock
// tokenizers pool, and a LongestPrefixScorer using the project's default
// backend weights.
func newTestIndexer(t *testing.T, tp kvblock.TokenProcessor, pool kvcache.TokenizersPool) *kvcache.Indexer {
	t.Helper()

	idx, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
	require.NoError(t, err)

	scorer, err := kvcache.NewKVBlockScorer(kvcache.DefaultKVBlockScorerConfig())
	require.NoError(t, err)

	return kvcache.NewIndexerForTest(tp, idx, scorer, pool)
}

// populateIndex inserts block-key -> pod entries into the index.
func populateIndex(t *testing.T, idx kvblock.Index, entries map[kvblock.BlockHash][]kvblock.PodEntry) {
	t.Helper()
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	for key, pods := range entries {
		err := idx.Add(ctx, []kvblock.BlockHash{key}, []kvblock.BlockHash{key}, pods)
		require.NoError(t, err)
	}
}

// --- scoring tests (shared scenarios) ---------------------------------------

// scoringTestCase defines a scenario exercised through both GetPodScores and
// ScoreTokens.
type scoringTestCase struct {
	name           string
	blockKeys      []uint64
	tokens         []uint32
	indexEntries   map[kvblock.BlockHash][]kvblock.PodEntry
	podIdentifiers []string
	wantScores     map[string]float64 // expected pod -> score (checked with InDelta)
	wantNil        bool               // if true, expect nil scores (not just empty)
}

var scoringTests = []scoringTestCase{
	{
		name:      "empty tokens",
		blockKeys: nil,
		tokens:    nil,
		wantNil:   true,
	},
	{
		name:       "no matching pods",
		blockKeys:  []uint64{100, 200, 300},
		tokens:     []uint32{1, 2, 3},
		wantScores: map[string]float64{},
	},
	{
		name:      "single pod full match",
		blockKeys: []uint64{10, 20, 30},
		tokens:    []uint32{1, 2, 3},
		indexEntries: map[kvblock.BlockHash][]kvblock.PodEntry{
			10: {{PodIdentifier: testPodA, DeviceTier: "gpu"}},
			20: {{PodIdentifier: testPodA, DeviceTier: "gpu"}},
			30: {{PodIdentifier: testPodA, DeviceTier: "gpu"}},
		},
		wantScores: map[string]float64{testPodA: 3.0},
	},
	{
		name:      "multiple pods",
		blockKeys: []uint64{10, 20, 30},
		tokens:    []uint32{1, 2, 3},
		indexEntries: map[kvblock.BlockHash][]kvblock.PodEntry{
			10: {
				{PodIdentifier: testPodA, DeviceTier: "gpu"},
				{PodIdentifier: testPodB, DeviceTier: "gpu"},
			},
			20: {
				{PodIdentifier: testPodA, DeviceTier: "gpu"},
				{PodIdentifier: testPodB, DeviceTier: "gpu"},
			},
			30: {
				{PodIdentifier: testPodA, DeviceTier: "gpu"},
			},
		},
		wantScores: map[string]float64{testPodA: 3.0, testPodB: 2.0},
	},
	{
		name:      "mixed device tiers",
		blockKeys: []uint64{10, 20},
		tokens:    []uint32{1, 2},
		indexEntries: map[kvblock.BlockHash][]kvblock.PodEntry{
			10: {{PodIdentifier: testPodA, DeviceTier: "gpu"}},
			20: {{PodIdentifier: testPodA, DeviceTier: "cpu"}},
		},
		wantScores: map[string]float64{testPodA: 1.8}, // gpu(1.0) + cpu(0.8)
	},
	{
		name:      "pod identifier filter",
		blockKeys: []uint64{10, 20},
		tokens:    []uint32{1, 2},
		indexEntries: map[kvblock.BlockHash][]kvblock.PodEntry{
			10: {
				{PodIdentifier: testPodA, DeviceTier: "gpu"},
				{PodIdentifier: testPodB, DeviceTier: "gpu"},
			},
			20: {
				{PodIdentifier: testPodA, DeviceTier: "gpu"},
				{PodIdentifier: testPodB, DeviceTier: "gpu"},
			},
		},
		podIdentifiers: []string{testPodA},
		wantScores:     map[string]float64{testPodA: 2.0},
	},
	{
		name:      "prefix break",
		blockKeys: []uint64{10, 20, 30},
		tokens:    []uint32{1, 2, 3},
		indexEntries: map[kvblock.BlockHash][]kvblock.PodEntry{
			10: {
				{PodIdentifier: testPodA, DeviceTier: "gpu"},
				{PodIdentifier: testPodB, DeviceTier: "gpu"},
			},
			20: {
				{PodIdentifier: testPodA, DeviceTier: "gpu"},
				// testPodB missing => prefix breaks for podB
			},
			30: {
				{PodIdentifier: testPodA, DeviceTier: "gpu"},
				{PodIdentifier: testPodB, DeviceTier: "gpu"},
			},
		},
		wantScores: map[string]float64{testPodA: 3.0, testPodB: 1.0},
	},
	{
		name:      "empty pod identifiers returns all",
		blockKeys: []uint64{10},
		tokens:    []uint32{1},
		indexEntries: map[kvblock.BlockHash][]kvblock.PodEntry{
			10: {
				{PodIdentifier: testPodA, DeviceTier: "gpu"},
				{PodIdentifier: testPodB, DeviceTier: "gpu"},
			},
		},
		podIdentifiers: []string{},
		wantScores:     map[string]float64{testPodA: 1.0, testPodB: 1.0},
	},
	{
		name:      "deterministic",
		blockKeys: []uint64{10, 20},
		tokens:    []uint32{42, 43},
		indexEntries: map[kvblock.BlockHash][]kvblock.PodEntry{
			10: {{PodIdentifier: testPodA, DeviceTier: "gpu"}},
			20: {{PodIdentifier: testPodA, DeviceTier: "gpu"}},
		},
		wantScores: map[string]float64{testPodA: 2.0},
	},
}

// assertScores verifies that the returned scores match expectations.
func assertScores(t *testing.T, tt *scoringTestCase, scores map[string]float64, err error) {
	t.Helper()
	require.NoError(t, err)

	if tt.wantNil {
		assert.Nil(t, scores, "expected nil scores")
		return
	}

	require.Len(t, scores, len(tt.wantScores), "unexpected number of scored pods")
	for pod, want := range tt.wantScores {
		require.Contains(t, scores, pod, "missing pod %q in scores", pod)
		assert.InDelta(t, want, scores[pod], 0.0001, "pod %q score mismatch", pod)
	}
}

func TestGetPodScores(t *testing.T) {
	for _, tt := range scoringTests {
		t.Run(tt.name, func(t *testing.T) {
			tp := &mockTokenProcessor{blockKeys: u64ToBlockKeys(tt.blockKeys)}
			pool := &mockTokenizersPool{tokens: tt.tokens}
			indexer := newTestIndexer(t, tp, pool)

			ctx := logging.NewTestLoggerIntoContext(context.Background())
			if tt.indexEntries != nil {
				populateIndex(t, indexer.KVBlockIndex(), tt.indexEntries)
			}

			scores, err := indexer.GetPodScores(ctx, nil, "hello", testModel, tt.podIdentifiers)
			assertScores(t, &tt, scores, err)
		})
	}
}

func TestScoreTokens(t *testing.T) {
	for _, tt := range scoringTests {
		t.Run(tt.name, func(t *testing.T) {
			tp := &mockTokenProcessor{blockKeys: u64ToBlockKeys(tt.blockKeys)}
			indexer := newTestIndexer(t, tp, &mockTokenizersPool{})

			ctx := logging.NewTestLoggerIntoContext(context.Background())
			if tt.indexEntries != nil {
				populateIndex(t, indexer.KVBlockIndex(), tt.indexEntries)
			}

			scores, err := indexer.ScoreTokens(ctx, tt.tokens, testModel, tt.podIdentifiers, nil)
			assertScores(t, &tt, scores, err)
		})
	}
}

// --- ScoreTokens storage tests -----------------------------------------------
// These cover the storage checkpoint scoring path added by Phase 4.

// newTestIndexerWithStorage creates an Indexer backed by an in-memory GPU index,
// a real CuckooStorageIndex, and the provided storage config.
func newTestIndexerWithStorage(t *testing.T, tp kvblock.TokenProcessor, storageCfg *kvcache.StorageConfig) (*kvcache.Indexer, *kvblock.CuckooStorageIndex) {
	t.Helper()

	gpuIndex, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
	require.NoError(t, err)

	scorer, err := kvcache.NewKVBlockScorer(kvcache.DefaultKVBlockScorerConfig())
	require.NoError(t, err)

	storageIndex, err := kvblock.NewCuckooStorageIndex(nil)
	require.NoError(t, err)

	indexer := kvcache.NewIndexerForTestWithStorage(tp, gpuIndex, scorer, &mockTokenizersPool{}, storageIndex, storageCfg)
	return indexer, storageIndex
}

func TestScoreTokens_StorageBonusApplied(t *testing.T) {
	// Checkpoints are sampled from canonical blockKeys at stride intervals.
	//
	// Setup:
	//   blockKeys: [10, 20, 30, 40, 50, 60] — stride=2, checkpoints at indices 1,3,5
	//   checkpoint keys: [20, 40, 60]
	//   GPU index: pod-a has blocks 10 and 20 (GPU score = 2.0)
	//   Storage index: has checkpoint key 20 (index 0), misses 40 (index 1) → walk stops
	//
	// Expected:
	//   highestCheckpoint = 0
	//   storagePrefixBlocks = (0+1) * 2 = 2
	//   2 > 2? No → no bonus. GPU score = 2.0
	//
	// To get a bonus, we need storagePrefixBlocks > gpuPrefixLen.
	// Use stride=1 so every block is a checkpoint.
	//   blockKeys: [10, 20, 30, 40] — stride=1, checkpoints at 0,1,2,3 → keys [10,20,30,40]
	//   Storage has all 4 checkpoints → highestCheckpoint=3
	//   storagePrefixBlocks = (3+1)*1 = 4
	//   GPU: pod-a has blocks 10,20 (score=2.0)
	//   4 > 2 → bonus = (4-2)*0.3 = 0.6
	//   final = 2.0 + 0.6 = 2.6

	tp := &mockTokenProcessor{
		blockKeys: u64ToBlockKeys([]uint64{10, 20, 30, 40}),
	}

	storageCfg := &kvcache.StorageConfig{
		CheckpointStride: 1,
		StorageWeight:    0.3,
		MinPrefixBlocks:  1,
	}

	indexer, storageIndex := newTestIndexerWithStorage(t, tp, storageCfg)
	ctx := logging.NewTestLoggerIntoContext(context.Background())

	// GPU: pod-a has blocks 10 and 20 (prefix of length 2)
	populateIndex(t, indexer.KVBlockIndex(), map[kvblock.BlockHash][]kvblock.PodEntry{
		10: {{PodIdentifier: testPodA, DeviceTier: "gpu"}},
		20: {{PodIdentifier: testPodA, DeviceTier: "gpu"}},
	})

	// Storage: all 4 canonical keys present as checkpoints
	storageEntries := []kvblock.PodEntry{{PodIdentifier: "shared-storage", DeviceTier: "storage"}}
	err := storageIndex.Add(ctx, nil,
		u64ToBlockKeys([]uint64{10, 20, 30, 40}),
		storageEntries)
	require.NoError(t, err)

	scores, err := indexer.ScoreTokens(ctx, []uint32{1, 2, 3, 4}, testModel, nil, nil)
	require.NoError(t, err)
	require.Contains(t, scores, testPodA)
	assert.InDelta(t, 2.6, scores[testPodA], 0.0001,
		"GPU score (2.0) + storage bonus ((4-2)*0.3 = 0.6) = 2.6")
}

func TestScoreTokens_StorageWithinGPUPrefix_NoBonus(t *testing.T) {
	// GPU covers more than storage → no bonus applied.
	//
	// blockKeys: [10, 20, 30] — stride=1, checkpoints at 0,1,2 → keys [10,20,30]
	// GPU: pod-a has all 3 blocks (GPU score = 3.0)
	// Storage: only checkpoint key 10 → highestCheckpoint=0
	// storagePrefixBlocks = (0+1)*1 = 1. 1 < 3 → no bonus.

	tp := &mockTokenProcessor{
		blockKeys: u64ToBlockKeys([]uint64{10, 20, 30}),
	}

	storageCfg := &kvcache.StorageConfig{
		CheckpointStride: 1,
		StorageWeight:    0.3,
		MinPrefixBlocks:  1,
	}

	indexer, storageIndex := newTestIndexerWithStorage(t, tp, storageCfg)
	ctx := logging.NewTestLoggerIntoContext(context.Background())

	populateIndex(t, indexer.KVBlockIndex(), map[kvblock.BlockHash][]kvblock.PodEntry{
		10: {{PodIdentifier: testPodA, DeviceTier: "gpu"}},
		20: {{PodIdentifier: testPodA, DeviceTier: "gpu"}},
		30: {{PodIdentifier: testPodA, DeviceTier: "gpu"}},
	})

	storageEntries := []kvblock.PodEntry{{PodIdentifier: "shared-storage", DeviceTier: "storage"}}
	err := storageIndex.Add(ctx, nil, []kvblock.BlockHash{kvblock.BlockHash(10)}, storageEntries)
	require.NoError(t, err)

	scores, err := indexer.ScoreTokens(ctx, []uint32{1, 2, 3}, testModel, nil, nil)
	require.NoError(t, err)
	require.Contains(t, scores, testPodA)
	assert.InDelta(t, 3.0, scores[testPodA], 0.0001,
		"storage covers less than GPU prefix — no bonus applied")
}

func TestScoreTokens_StorageBelowMinPrefixBlocks_NoBonus(t *testing.T) {
	// Storage coverage exists but is below MinPrefixBlocks threshold.
	//
	// blockKeys: [10, 20, 30, 40] — stride=2, checkpoints at indices 1,3 → keys [20,40]
	// Storage has key 20 → highestCheckpoint=0
	// storagePrefixBlocks = (0+1)*2 = 2. MinPrefixBlocks=100 → 2 < 100 → no bonus.

	tp := &mockTokenProcessor{
		blockKeys: u64ToBlockKeys([]uint64{10, 20, 30, 40}),
	}

	storageCfg := &kvcache.StorageConfig{
		CheckpointStride: 2,
		StorageWeight:    0.3,
		MinPrefixBlocks:  100, // threshold much higher than coverage
	}

	indexer, storageIndex := newTestIndexerWithStorage(t, tp, storageCfg)
	ctx := logging.NewTestLoggerIntoContext(context.Background())

	populateIndex(t, indexer.KVBlockIndex(), map[kvblock.BlockHash][]kvblock.PodEntry{
		10: {{PodIdentifier: testPodA, DeviceTier: "gpu"}},
	})

	storageEntries := []kvblock.PodEntry{{PodIdentifier: "shared-storage", DeviceTier: "storage"}}
	err := storageIndex.Add(ctx, nil, []kvblock.BlockHash{kvblock.BlockHash(20)}, storageEntries)
	require.NoError(t, err)

	scores, err := indexer.ScoreTokens(ctx, []uint32{1, 2, 3, 4}, testModel, nil, nil)
	require.NoError(t, err)
	require.Contains(t, scores, testPodA)
	assert.InDelta(t, 1.0, scores[testPodA], 0.0001,
		"storage below MinPrefixBlocks — no bonus applied")
}

func TestScoreTokens_DifferentialBonusAcrossPods(t *testing.T) {
	// Two pods with different GPU prefix lengths get different storage bonuses.
	//
	// blockKeys: [10, 20, 30, 40, 50, 60] — stride=2, checkpoints at 1,3,5 → keys [20,40,60]
	// GPU: pod-a has 10,20,30 (score=3.0), pod-b has 10 (score=1.0)
	// Storage: all 3 checkpoints present → highestCheckpoint=2
	// storagePrefixBlocks = (2+1)*2 = 6
	//
	// pod-a: 6 > 3 → bonus = (6-3)*0.3 = 0.9, final = 3.9
	// pod-b: 6 > 1 → bonus = (6-1)*0.3 = 1.5, final = 2.5

	tp := &mockTokenProcessor{
		blockKeys: u64ToBlockKeys([]uint64{10, 20, 30, 40, 50, 60}),
	}

	storageCfg := &kvcache.StorageConfig{
		CheckpointStride: 2,
		StorageWeight:    0.3,
		MinPrefixBlocks:  1,
	}

	indexer, storageIndex := newTestIndexerWithStorage(t, tp, storageCfg)
	ctx := logging.NewTestLoggerIntoContext(context.Background())

	populateIndex(t, indexer.KVBlockIndex(), map[kvblock.BlockHash][]kvblock.PodEntry{
		10: {
			{PodIdentifier: testPodA, DeviceTier: "gpu"},
			{PodIdentifier: testPodB, DeviceTier: "gpu"},
		},
		20: {{PodIdentifier: testPodA, DeviceTier: "gpu"}},
		30: {{PodIdentifier: testPodA, DeviceTier: "gpu"}},
	})

	storageEntries := []kvblock.PodEntry{{PodIdentifier: "shared-storage", DeviceTier: "storage"}}
	err := storageIndex.Add(ctx, nil,
		u64ToBlockKeys([]uint64{20, 40, 60}),
		storageEntries)
	require.NoError(t, err)

	scores, err := indexer.ScoreTokens(ctx, []uint32{1, 2, 3, 4, 5, 6}, testModel, nil, nil)
	require.NoError(t, err)

	require.Contains(t, scores, testPodA)
	require.Contains(t, scores, testPodB)
	assert.InDelta(t, 3.9, scores[testPodA], 0.0001,
		"pod-a: GPU(3.0) + storage((6-3)*0.3=0.9) = 3.9")
	assert.InDelta(t, 2.5, scores[testPodB], 0.0001,
		"pod-b: GPU(1.0) + storage((6-1)*0.3=1.5) = 2.5")
}

func TestScoreTokens_NoStorageIndex_Unchanged(t *testing.T) {
	// Verify that nil storageIndex produces identical results to the base scoring path.
	tp := &mockTokenProcessor{
		blockKeys: u64ToBlockKeys([]uint64{10, 20, 30}),
	}

	indexer := newTestIndexer(t, tp, &mockTokenizersPool{})
	ctx := logging.NewTestLoggerIntoContext(context.Background())

	populateIndex(t, indexer.KVBlockIndex(), map[kvblock.BlockHash][]kvblock.PodEntry{
		10: {{PodIdentifier: testPodA, DeviceTier: "gpu"}},
		20: {{PodIdentifier: testPodA, DeviceTier: "gpu"}},
	})

	scores, err := indexer.ScoreTokens(ctx, []uint32{1, 2, 3}, testModel, nil, nil)
	require.NoError(t, err)
	require.Contains(t, scores, testPodA)
	assert.InDelta(t, 2.0, scores[testPodA], 0.0001,
		"no storage index — GPU-only score unchanged")
}

func TestScoreTokens_StorageCheckpointGap_StopsAtMiss(t *testing.T) {
	// Walk stops at the first missing checkpoint — later hits are ignored.
	//
	// blockKeys: [10, 20, 30, 40, 50, 60] — stride=2, checkpoints at 1,3,5 → keys [20,40,60]
	// Storage has keys 20 and 60 but NOT 40 (gap at checkpoint index 1)
	// Walk: 20 hit, 40 miss → stop. highestCheckpoint = 0
	// storagePrefixBlocks = (0+1)*2 = 2
	// GPU: pod-a has block 10 (score=1.0)
	// 2 > 1 → bonus = (2-1)*0.3 = 0.3, final = 1.3

	tp := &mockTokenProcessor{
		blockKeys: u64ToBlockKeys([]uint64{10, 20, 30, 40, 50, 60}),
	}

	storageCfg := &kvcache.StorageConfig{
		CheckpointStride: 2,
		StorageWeight:    0.3,
		MinPrefixBlocks:  1,
	}

	indexer, storageIndex := newTestIndexerWithStorage(t, tp, storageCfg)
	ctx := logging.NewTestLoggerIntoContext(context.Background())

	populateIndex(t, indexer.KVBlockIndex(), map[kvblock.BlockHash][]kvblock.PodEntry{
		10: {{PodIdentifier: testPodA, DeviceTier: "gpu"}},
	})

	// Insert checkpoints 0 and 2, but NOT checkpoint 1 (gap)
	storageEntries := []kvblock.PodEntry{{PodIdentifier: "shared-storage", DeviceTier: "storage"}}
	err := storageIndex.Add(ctx, nil,
		[]kvblock.BlockHash{kvblock.BlockHash(20), kvblock.BlockHash(60)},
		storageEntries)
	require.NoError(t, err)

	scores, err := indexer.ScoreTokens(ctx, []uint32{1, 2, 3, 4, 5, 6}, testModel, nil, nil)
	require.NoError(t, err)
	require.Contains(t, scores, testPodA)
	assert.InDelta(t, 1.3, scores[testPodA], 0.0001,
		"walk stops at gap: GPU(1.0) + storage((2-1)*0.3=0.3) = 1.3")
}

// --- GetPodScores-specific tests --------------------------------------------
// These cover behavior unique to GetPodScores that ScoreTokens
// does not have (i.e. prompt truncation).

func TestGetPodScores_TruncatePromptTokens(t *testing.T) {
	// The mock pool returns 5 tokens. With TruncatePromptTokens=3, only
	// the last 3 tokens (300, 400, 500) should be passed to the token
	// processor. We verify this via tp.receivedTokens.
	blockKeys := u64ToBlockKeys([]uint64{10, 20, 30})
	tp := &mockTokenProcessor{blockKeys: blockKeys}
	pool := &mockTokenizersPool{tokens: []uint32{100, 200, 300, 400, 500}}
	indexer := newTestIndexer(t, tp, pool)

	ctx := logging.NewTestLoggerIntoContext(context.Background())
	populateIndex(t, indexer.KVBlockIndex(), map[kvblock.BlockHash][]kvblock.PodEntry{
		10: {{PodIdentifier: testPodA, DeviceTier: "gpu"}},
		20: {{PodIdentifier: testPodA, DeviceTier: "gpu"}},
		30: {{PodIdentifier: testPodA, DeviceTier: "gpu"}},
	})

	truncateLimit := 3
	renderReq := &types.RenderChatRequest{
		TruncatePromptTokens: &truncateLimit,
	}

	scores, err := indexer.GetPodScores(ctx, renderReq, "", testModel, nil)
	require.NoError(t, err)
	require.Contains(t, scores, testPodA)
	assert.InDelta(t, 3.0, scores[testPodA], 0.0001)
	assert.Equal(t, []uint32{300, 400, 500}, tp.receivedTokens,
		"token processor should receive only the last 3 tokens after truncation")
}

func TestGetPodScores_TruncateNoOp(t *testing.T) {
	// TruncatePromptTokens is set but larger than the token count — no
	// truncation should happen, all tokens are passed through.
	blockKeys := u64ToBlockKeys([]uint64{10, 20})
	tp := &mockTokenProcessor{blockKeys: blockKeys}
	pool := &mockTokenizersPool{tokens: []uint32{1, 2}}
	indexer := newTestIndexer(t, tp, pool)

	ctx := logging.NewTestLoggerIntoContext(context.Background())
	populateIndex(t, indexer.KVBlockIndex(), map[kvblock.BlockHash][]kvblock.PodEntry{
		10: {{PodIdentifier: testPodA, DeviceTier: "gpu"}},
		20: {{PodIdentifier: testPodA, DeviceTier: "gpu"}},
	})

	truncateLimit := 100 // larger than token count
	renderReq := &types.RenderChatRequest{
		TruncatePromptTokens: &truncateLimit,
	}

	scores, err := indexer.GetPodScores(ctx, renderReq, "", testModel, nil)
	require.NoError(t, err)
	require.Contains(t, scores, testPodA)
	assert.InDelta(t, 2.0, scores[testPodA], 0.0001)
	assert.Equal(t, []uint32{1, 2}, tp.receivedTokens,
		"token processor should receive all tokens when limit exceeds count")
}

func TestGetPodScores_TruncateZero(t *testing.T) {
	// TruncatePromptTokens=0 should not truncate (the code checks limit > 0).
	blockKeys := u64ToBlockKeys([]uint64{10, 20})
	tp := &mockTokenProcessor{blockKeys: blockKeys}
	pool := &mockTokenizersPool{tokens: []uint32{1, 2}}
	indexer := newTestIndexer(t, tp, pool)

	ctx := logging.NewTestLoggerIntoContext(context.Background())
	populateIndex(t, indexer.KVBlockIndex(), map[kvblock.BlockHash][]kvblock.PodEntry{
		10: {{PodIdentifier: testPodA, DeviceTier: "gpu"}},
		20: {{PodIdentifier: testPodA, DeviceTier: "gpu"}},
	})

	truncateLimit := 0
	renderReq := &types.RenderChatRequest{
		TruncatePromptTokens: &truncateLimit,
	}

	scores, err := indexer.GetPodScores(ctx, renderReq, "", testModel, nil)
	require.NoError(t, err)
	require.Contains(t, scores, testPodA)
	assert.InDelta(t, 2.0, scores[testPodA], 0.0001, "zero limit should not truncate")
	assert.Equal(t, []uint32{1, 2}, tp.receivedTokens,
		"token processor should receive all tokens when limit is zero")
}
