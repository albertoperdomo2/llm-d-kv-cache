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

package helper

import (
	"context"
	"net"
	"os"
	"strconv"

	"github.com/llm-d/llm-d-kv-cache/examples/testdata"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	// EnvTokenizerEndpoint is the env var for the UDS tokenizer socket path or TCP address.
	// Use a path (e.g. /tmp/tokenizer/tokenizer-uds.socket) for UDS mode,
	// or host:port (e.g. localhost:50051) for TCP mode.
	EnvTokenizerEndpoint = "TOKENIZER_ENDPOINT" //nolint:gosec // env var name, not a credential

	envStorageIndexEnabled    = "STORAGE_INDEX_ENABLED"
	envStorageBlockSize       = "STORAGE_BLOCK_SIZE"
	envCheckpointStride       = "STORAGE_CHECKPOINT_STRIDE"
	envStorageWeight          = "STORAGE_WEIGHT"
	envStorageMinPrefixBlocks = "STORAGE_MIN_PREFIX_BLOCKS"
)

func isTCPAddr(s string) bool {
	host, port, err := net.SplitHostPort(s)
	return err == nil && host != "" && port != ""
}

// ApplyTokenizerEndpoint reads TOKENIZER_ENDPOINT and sets UDS config on the given config.
func ApplyTokenizerEndpoint(config *kvcache.Config) {
	endpoint := os.Getenv(EnvTokenizerEndpoint)
	if endpoint == "" {
		return
	}
	config.TokenizersPoolConfig.UdsTokenizerConfig = &tokenization.UdsTokenizerConfig{
		SocketFile: endpoint,
		UseTCP:     isTCPAddr(endpoint),
	}
}

func getKVCacheIndexerConfig() (*kvcache.Config, error) {
	config, err := kvcache.NewDefaultConfig()
	if err != nil {
		return nil, err
	}

	config.TokenizersPoolConfig.ModelName = testdata.ModelName
	ApplyTokenizerEndpoint(config)
	applyStorageExampleConfig(config)

	return config, nil
}

func applyStorageExampleConfig(config *kvcache.Config) {
	if config == nil {
		return
	}

	storageCfg := config.StorageConfig
	if storageCfg == nil {
		storageCfg = kvcache.DefaultStorageConfig()
		config.StorageConfig = storageCfg
	}

	if enabled, err := strconv.ParseBool(os.Getenv(envStorageIndexEnabled)); err == nil {
		storageCfg.Enabled = enabled
	}
	if blockSize, err := strconv.Atoi(os.Getenv(envStorageBlockSize)); err == nil && blockSize > 0 {
		storageCfg.StorageBlockSize = blockSize
	}
	if stride, err := strconv.Atoi(os.Getenv(envCheckpointStride)); err == nil && stride > 0 {
		storageCfg.CheckpointStride = stride
	}
	if minPrefix, err := strconv.Atoi(os.Getenv(envStorageMinPrefixBlocks)); err == nil && minPrefix > 0 {
		storageCfg.MinPrefixBlocks = minPrefix
	}
	if weight, err := strconv.ParseFloat(os.Getenv(envStorageWeight), 64); err == nil && weight >= 0 {
		storageCfg.StorageWeight = weight
	}
}

func getTokenProcessorConfig() *kvblock.TokenProcessorConfig {
	return &kvblock.TokenProcessorConfig{
		BlockSize: 256,
	}
}

func SetupKVCacheIndexer(ctx context.Context) (*kvcache.Indexer, error) {
	logger := log.FromContext(ctx)

	cfg, err := getKVCacheIndexerConfig()
	if err != nil {
		return nil, err
	}

	tokenProcessorConfig := getTokenProcessorConfig()
	cfg.TokenProcessorConfig = tokenProcessorConfig

	tokenProcessor, err := kvblock.NewChunkedTokenDatabase(tokenProcessorConfig)
	if err != nil {
		return nil, err
	}

	kvCacheIndexer, err := kvcache.NewKVCacheIndexer(ctx, cfg, tokenProcessor)
	if err != nil {
		return nil, err
	}

	logger.Info("Created Indexer")

	go kvCacheIndexer.Run(ctx)
	logger.Info("Started Indexer", "model", testdata.ModelName)

	return kvCacheIndexer, nil
}
