# Changelog

## [0.5.1](https://github.com/finngi/mycelium/compare/oyster-v0.5.0...oyster-v0.5.1) (2026-07-09)


### Bug Fixes

* **reishi:** preserve unknown manifest keys; tolerate non-scalar board metrics; declare Alpha status ([#20](https://github.com/finngi/mycelium/issues/20)) ([3078a71](https://github.com/finngi/mycelium/commit/3078a71d2f17e22778cb7639fd9ef87f4c2a253d))

## [0.5.0](https://github.com/finngi/mycelium/compare/oyster-v0.4.1...oyster-v0.5.0) (2026-07-09)


### Features

* add real MLX LoRA trainer and typed trainer contract ([#8](https://github.com/finngi/mycelium/issues/8)) ([7a39b31](https://github.com/finngi/mycelium/commit/7a39b31755f2928f764c9bc6017389fcd8944f89))
* move mesh claims off main onto an auto-synced mesh-state branch ([#15](https://github.com/finngi/mycelium/issues/15)) ([a9869df](https://github.com/finngi/mycelium/commit/a9869df5d3346f6fb887d2cd6084d2cd72f79278))


### Bug Fixes

* **ci:** grant contents:read alongside security-events:write in codeql ([7ba6d9e](https://github.com/finngi/mycelium/commit/7ba6d9e3d044cb25600a7967e99e3844f369e6d4))
* **ci:** lint with uvx, not uv run, to avoid syncing project deps ([f666def](https://github.com/finngi/mycelium/commit/f666def4c64d608727f994da04c3387d79682a5d))
* **ci:** pass a token to cross-repo mcm-reishi checkouts ([0defd4f](https://github.com/finngi/mycelium/commit/0defd4ff0398f957a116886d6eaaf1a1ed5a3e6b))
* reset local state when a claim/heartbeat push is permanently rejected ([#12](https://github.com/finngi/mycelium/issues/12)) ([85be8dc](https://github.com/finngi/mycelium/commit/85be8dcd150aaf522b5c833f56438dc6764e6d6c))


### Documentation

* add NOTICE copyright + package authors ([78da29f](https://github.com/finngi/mycelium/commit/78da29f169d64ae1100098b696b38d4a7daaff43))
* **oyster:** trim docstrings that narrated other modules' behavior ([b4a0bad](https://github.com/finngi/mycelium/commit/b4a0bad87792d4cdc1897f206c7d92af05afcba6))
* point cross-package links at the mycelium monorepo ([baea95c](https://github.com/finngi/mycelium/commit/baea95ca4fa34d9540d4b9aa595a9cb019bd0c82))
* set package authors to Finn Danger Cathersides ([a70a2ca](https://github.com/finngi/mycelium/commit/a70a2cada573735a003e52aa89b9f82e5a0cbb6d))

## [0.4.1](https://github.com/finngi/mycelium/compare/oyster-v0.4.0...oyster-v0.4.1) (2026-07-09)


### Documentation

* add NOTICE copyright + package authors ([60714a8](https://github.com/finngi/mycelium/commit/60714a8524504b5c2b374bac0fa2ddbca899ae46))
* **oyster:** trim docstrings that narrated other modules' behavior ([12ac13c](https://github.com/finngi/mycelium/commit/12ac13c5cb5e21a8536ac05ed9818c786c28fd24))
* set package authors to Finn Danger Cathersides ([c2a52a6](https://github.com/finngi/mycelium/commit/c2a52a6aa57c9e0b6204e3ca56d752986a88bc61))

## [0.4.0](https://github.com/finngi/mycelium/compare/oyster-v0.3.0...oyster-v0.4.0) (2026-07-09)


### Features

* add real MLX LoRA trainer and typed trainer contract ([#8](https://github.com/finngi/mycelium/issues/8)) ([1562b3f](https://github.com/finngi/mycelium/commit/1562b3f1e3a6f25b16f2afdbe8587125aa12c851))
* move mesh claims off main onto an auto-synced mesh-state branch ([#15](https://github.com/finngi/mycelium/issues/15)) ([28f4f97](https://github.com/finngi/mycelium/commit/28f4f97af419b718bc3d656d8898103b720f7002))


### Bug Fixes

* **ci:** grant contents:read alongside security-events:write in codeql ([e9ce5f8](https://github.com/finngi/mycelium/commit/e9ce5f810b5fec193224a9e4761e7697e2f0254a))
* **ci:** lint with uvx, not uv run, to avoid syncing project deps ([d2039fc](https://github.com/finngi/mycelium/commit/d2039fc3cdb6c155b11428cfcb3938ac5df7bef5))
* **ci:** pass a token to cross-repo mcm-reishi checkouts ([0672ba2](https://github.com/finngi/mycelium/commit/0672ba23b45d2ba97988d4373b2c0f50bc200bb2))
* reset local state when a claim/heartbeat push is permanently rejected ([#12](https://github.com/finngi/mycelium/issues/12)) ([1e8d28d](https://github.com/finngi/mycelium/commit/1e8d28d0d2c04b8ac44ecebfb9779f7002e23fb2))


### Documentation

* point cross-package links at the mycelium monorepo ([2ec0bb9](https://github.com/finngi/mycelium/commit/2ec0bb924c9003385c4d2e6ce266db5a0951f634))

## [0.3.0](https://github.com/finngi/mcm-oyster/compare/v0.2.1...v0.3.0) (2026-07-06)


### Features

* move mesh claims off main onto an auto-synced mesh-state branch ([#15](https://github.com/finngi/mcm-oyster/issues/15)) ([9511f81](https://github.com/finngi/mcm-oyster/commit/9511f81cd97061239cab61156d85288742c4e079))

## [0.2.1](https://github.com/finngi/mcm-oyster/compare/v0.2.0...v0.2.1) (2026-07-06)


### Bug Fixes

* reset local state when a claim/heartbeat push is permanently rejected ([#12](https://github.com/finngi/mcm-oyster/issues/12)) ([b2e4c88](https://github.com/finngi/mcm-oyster/commit/b2e4c8807a92a3e6b028346cdd7db8c66d765730))

## [0.2.0](https://github.com/finngi/mcm-oyster/compare/v0.1.1...v0.2.0) (2026-07-06)


### Features

* add real MLX LoRA trainer and typed trainer contract ([#8](https://github.com/finngi/mcm-oyster/issues/8)) ([786b8b9](https://github.com/finngi/mcm-oyster/commit/786b8b9db483975699648d84bd9b09ad984c8301))

## [0.1.1](https://github.com/finngi/mcm-oyster/compare/v0.1.0...v0.1.1) (2026-07-05)


### Bug Fixes

* **ci:** grant contents:read alongside security-events:write in codeql ([73f1492](https://github.com/finngi/mcm-oyster/commit/73f149252bc48ab1b76cf6b33ae5221a04a425db))
* **ci:** lint with uvx, not uv run, to avoid syncing project deps ([711a8a7](https://github.com/finngi/mcm-oyster/commit/711a8a7af56bec8e425951ef614aa373eb67f275))
* **ci:** pass a token to cross-repo mcm-reishi checkouts ([9bea895](https://github.com/finngi/mcm-oyster/commit/9bea8955807a7f21373d64b8d79fd439405c0f97))
