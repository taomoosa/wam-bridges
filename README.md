# WAM Bridges

Practical examples for adapting world action models to new robot embodiments.
Each example uses an upstream framework and defines its dataset, configuration,
and inference contract.

| Example | Scope |
| --- | --- |
| [Cosmos3](bridges/cosmos3/README.md) | Nano SFT and inference for single/dual arms and other joint-position mechanisms, up to 64 channels |

Copy the example files into the upstream project manually. Weights, datasets,
environments, and robot controllers are supplied separately. Documentation and
code comments are in English.

## Licenses

Original bridge code and documentation use the [MIT license](LICENSE).
NVIDIA-derived files retain **OpenMDW-1.1**; their exact scope is listed in
[NOTICE](NOTICE). MIT does not relicense those files or external assets.

| External asset | License / source |
| --- | --- |
| Cosmos / cosmos-framework source | [OpenMDW-1.1](bridges/cosmos3/licenses/cosmos-framework-LICENSE) |
| Cosmos3-Nano, Nano-Policy-DROID | OpenMDW-1.1: [base](https://huggingface.co/nvidia/Cosmos3-Nano#license), [policy](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID#license) |
| Cosmos3-Edge-Policy-DROID | [OpenMDW-1.1](https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID#license) |
| Wan2.2 VAE | [Apache-2.0](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B#license-agreement) |
| LeRobot library | [Apache-2.0](https://github.com/mli0603/lerobot/blob/1a4316c6845330bc552fb982dbc44bdb4f66f2f1/LICENSE) |

Reviewed September 13, 2026: [OpenMDW-1.1](https://openmdw.ai/license/1-1/)
permits redistribution subject to its terms, including retaining its license and
applicable origin notices. This source-only example includes those notices and
license text. No weights, datasets, CUDA components, or container images are
redistributed. Check the terms of the exact assets/environment obtained separately;
NVIDIA products do not all share one license.
