# Changelog

## [0.6.1](https://github.com/finngi/mycelium/compare/reishi-v0.6.0...reishi-v0.6.1) (2026-07-09)


### Bug Fixes

* **reishi:** preserve unknown manifest keys; tolerate non-scalar board metrics; declare Alpha status ([#20](https://github.com/finngi/mycelium/issues/20)) ([3078a71](https://github.com/finngi/mycelium/commit/3078a71d2f17e22778cb7639fd9ef87f4c2a253d))

## [0.6.0](https://github.com/finngi/mycelium/compare/reishi-v0.5.1...reishi-v0.6.0) (2026-07-09)


### Features

* add TypedDict manifests, Task.aggregate scoring, and JSON codec ([#8](https://github.com/finngi/mycelium/issues/8)) ([e64ce5a](https://github.com/finngi/mycelium/commit/e64ce5a889c76dd9d55ee76a4dfa1ec921f8a73b))
* **store:** default to a sqlite manifest backend, add artifact root ([#13](https://github.com/finngi/mycelium/issues/13)) ([c337980](https://github.com/finngi/mycelium/commit/c337980f64d63862ef3c7342476838381c886630))
* **task:** generalize Task to any f(x)=y + pure eval seam ([#19](https://github.com/finngi/mycelium/issues/19)) ([044a35f](https://github.com/finngi/mycelium/commit/044a35f7e6351f46282f6101ea7cb25599094646))
* **tasks:** reishi ships no tasks; deployments load them via mcm.tasks entry points ([#16](https://github.com/finngi/mycelium/issues/16)) ([5c95655](https://github.com/finngi/mycelium/commit/5c95655f873ba93fb59c8631039e8e44c026bc8e))


### Bug Fixes

* **ci:** check out mcm-enoki as a sibling for experiment_submit tests ([242572a](https://github.com/finngi/mycelium/commit/242572ac54bb538c5a3f3727d9682da19433320e))
* **ci:** grant contents:read alongside security-events:write in codeql ([da56e13](https://github.com/finngi/mycelium/commit/da56e13be5400a4ad84cc5f2abcedb407b5dd322))
* **ci:** lint with uvx, not uv run, to avoid syncing project deps ([76cf12a](https://github.com/finngi/mycelium/commit/76cf12af7f28b9d2bee07753a2057457802cf6b2))
* **cli:** fail loud and clean on task-loading errors; isolate test registry ([#18](https://github.com/finngi/mycelium/issues/18)) ([7c6c8fe](https://github.com/finngi/mycelium/commit/7c6c8fec0512043e21dc1e227858a634d9ae0818))
* **store:** sqlite root() returns the store dir, not the db file ([ef82f48](https://github.com/finngi/mycelium/commit/ef82f48bb6e7bc9380ddaae6f11ed76c00436d63))


### Documentation

* add NOTICE copyright + package authors ([78da29f](https://github.com/finngi/mycelium/commit/78da29f169d64ae1100098b696b38d4a7daaff43))
* fix stale/inaccurate doc claims and comment restatements ([#11](https://github.com/finngi/mycelium/issues/11)) ([880f2e0](https://github.com/finngi/mycelium/commit/880f2e0cdb20f664dcb8d2475c2b8f52acac53ea))
* point cross-package links at the mycelium monorepo ([baea95c](https://github.com/finngi/mycelium/commit/baea95ca4fa34d9540d4b9aa595a9cb019bd0c82))
* **primitives:** rewrite comments/docstrings to actual scope ([621ce90](https://github.com/finngi/mycelium/commit/621ce90f8acb973f1fccf6c06eb9b5806844a816))
* **reishi:** rewrite non-primitive comments to actual scope ([4c3961e](https://github.com/finngi/mycelium/commit/4c3961edce7d1dafd08b9040601e3b444fbdeff3))
* set package authors to Finn Danger Cathersides ([a70a2ca](https://github.com/finngi/mycelium/commit/a70a2cada573735a003e52aa89b9f82e5a0cbb6d))

## [0.5.1](https://github.com/finngi/mycelium/compare/reishi-v0.5.0...reishi-v0.5.1) (2026-07-09)


### Documentation

* add NOTICE copyright + package authors ([60714a8](https://github.com/finngi/mycelium/commit/60714a8524504b5c2b374bac0fa2ddbca899ae46))
* **primitives:** rewrite comments/docstrings to actual scope ([b5127c4](https://github.com/finngi/mycelium/commit/b5127c4631b8e074579c424fc70067a8e2f7187d))
* **reishi:** rewrite non-primitive comments to actual scope ([c7c6817](https://github.com/finngi/mycelium/commit/c7c6817e802f6f6d56413d3e00847c03fb1456b4))
* set package authors to Finn Danger Cathersides ([c2a52a6](https://github.com/finngi/mycelium/commit/c2a52a6aa57c9e0b6204e3ca56d752986a88bc61))

## [0.5.0](https://github.com/finngi/mycelium/compare/reishi-v0.4.1...reishi-v0.5.0) (2026-07-09)


### Features

* add TypedDict manifests, Task.aggregate scoring, and JSON codec ([#8](https://github.com/finngi/mycelium/issues/8)) ([f700700](https://github.com/finngi/mycelium/commit/f700700c418170be5a9f90a7f8b507f14454b536))
* **store:** default to a sqlite manifest backend, add artifact root ([#13](https://github.com/finngi/mycelium/issues/13)) ([d7d570b](https://github.com/finngi/mycelium/commit/d7d570bbb84d2774151be7e4b49579488dfe4e11))
* **task:** generalize Task to any f(x)=y + pure eval seam ([#19](https://github.com/finngi/mycelium/issues/19)) ([cabdcfb](https://github.com/finngi/mycelium/commit/cabdcfb3a25794df1bc72f87ef8a07324e42b19f))
* **tasks:** reishi ships no tasks; deployments load them via mcm.tasks entry points ([#16](https://github.com/finngi/mycelium/issues/16)) ([db3f238](https://github.com/finngi/mycelium/commit/db3f2381ad95c07083060ef5a8b9b631b2e4ad94))


### Bug Fixes

* **ci:** check out mcm-enoki as a sibling for experiment_submit tests ([55ff3dc](https://github.com/finngi/mycelium/commit/55ff3dc4b85b2c395bd91ecb406de0a1d954fa85))
* **ci:** grant contents:read alongside security-events:write in codeql ([87d45d3](https://github.com/finngi/mycelium/commit/87d45d3e3c97c7abd4cbfa0c1425e391c58d5672))
* **ci:** lint with uvx, not uv run, to avoid syncing project deps ([d2c1a09](https://github.com/finngi/mycelium/commit/d2c1a0943790b6af19b338d3169a641d03f620b0))
* **cli:** fail loud and clean on task-loading errors; isolate test registry ([#18](https://github.com/finngi/mycelium/issues/18)) ([a6c8944](https://github.com/finngi/mycelium/commit/a6c89443e7beb89b820c977f2a43138e31c44049))
* **store:** sqlite root() returns the store dir, not the db file ([22dd3d4](https://github.com/finngi/mycelium/commit/22dd3d4af2ebea8991f07d328cc8388b27dd7d66))


### Documentation

* fix stale/inaccurate doc claims and comment restatements ([#11](https://github.com/finngi/mycelium/issues/11)) ([dce7420](https://github.com/finngi/mycelium/commit/dce74207a1376f311b2f28008f6494c5adb90cab))
* point cross-package links at the mycelium monorepo ([2ec0bb9](https://github.com/finngi/mycelium/commit/2ec0bb924c9003385c4d2e6ce266db5a0951f634))

## [0.4.1](https://github.com/finngi/mcm-reishi/compare/v0.4.0...v0.4.1) (2026-07-07)


### Bug Fixes

* **cli:** fail loud and clean on task-loading errors; isolate test registry ([#18](https://github.com/finngi/mcm-reishi/issues/18)) ([e13d540](https://github.com/finngi/mcm-reishi/commit/e13d54095dee07ec44394fd08700e05a2e5dde97))

## [0.4.0](https://github.com/finngi/mcm-reishi/compare/v0.3.0...v0.4.0) (2026-07-07)


### Features

* **tasks:** reishi ships no tasks; deployments load them via mcm.tasks entry points ([#16](https://github.com/finngi/mcm-reishi/issues/16)) ([d8e4e7a](https://github.com/finngi/mcm-reishi/commit/d8e4e7a8dbabdb6c8ac1bf77c34d8b3bd1d2e1ad))

## [0.3.0](https://github.com/finngi/mcm-reishi/compare/v0.2.0...v0.3.0) (2026-07-06)


### Features

* **store:** default to a sqlite manifest backend, add artifact root ([#13](https://github.com/finngi/mcm-reishi/issues/13)) ([841ea34](https://github.com/finngi/mcm-reishi/commit/841ea346767cff0fad82cbfd299f0a792fdd2c89))

## [0.2.0](https://github.com/finngi/mcm-reishi/compare/v0.1.1...v0.2.0) (2026-07-06)


### Features

* add TypedDict manifests, Task.aggregate scoring, and JSON codec ([#8](https://github.com/finngi/mcm-reishi/issues/8)) ([9f3e3ea](https://github.com/finngi/mcm-reishi/commit/9f3e3ea6ec10f43b744383b65a0c7c1a9522aca4))


### Documentation

* fix stale/inaccurate doc claims and comment restatements ([#11](https://github.com/finngi/mcm-reishi/issues/11)) ([32a15fa](https://github.com/finngi/mcm-reishi/commit/32a15faef13c98622f165c8ebb1e0c302dd65ac7))

## [0.1.1](https://github.com/finngi/mcm-reishi/compare/v0.1.0...v0.1.1) (2026-07-05)


### Bug Fixes

* **ci:** check out mcm-enoki as a sibling for experiment_submit tests ([8fd00af](https://github.com/finngi/mcm-reishi/commit/8fd00af277d92d71680cececaa163366c078aaf5))
* **ci:** grant contents:read alongside security-events:write in codeql ([a1f35c0](https://github.com/finngi/mcm-reishi/commit/a1f35c071b2ef0a6258a7bacaafe02ce346da134))
* **ci:** lint with uvx, not uv run, to avoid syncing project deps ([df77a50](https://github.com/finngi/mcm-reishi/commit/df77a50256217b7822555c6f392aad05896eb7d5))
