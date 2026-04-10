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

package kvcache

import "testing"

func TestValidateStorageConfig(t *testing.T) {
	t.Run("disabled config is allowed", func(t *testing.T) {
		cfg := DefaultStorageConfig()
		if err := validateStorageConfig(cfg, 16); err != nil {
			t.Fatalf("expected disabled storage config to validate, got %v", err)
		}
	})

	t.Run("enabled config validates", func(t *testing.T) {
		cfg := DefaultStorageConfig()
		cfg.Enabled = true
		if err := validateStorageConfig(cfg, 16); err != nil {
			t.Fatalf("expected enabled storage config to validate, got %v", err)
		}
	})

	t.Run("storage block size must be multiple of gpu block size", func(t *testing.T) {
		cfg := DefaultStorageConfig()
		cfg.Enabled = true
		cfg.StorageBlockSize = 250
		if err := validateStorageConfig(cfg, 16); err == nil {
			t.Fatal("expected invalid storage block size to fail validation")
		}
	})

	t.Run("checkpoint stride must be positive", func(t *testing.T) {
		cfg := DefaultStorageConfig()
		cfg.Enabled = true
		cfg.CheckpointStride = 0
		if err := validateStorageConfig(cfg, 16); err == nil {
			t.Fatal("expected zero checkpoint stride to fail validation")
		}
	})
}
