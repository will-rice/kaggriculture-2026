# Third-Party Notices

## Kaggriculture engine port (yhay81/three-day-shop-router)

`src/kaggriculture/campaign/engine/sim.hpp` and
`src/kaggriculture/campaign/engine/pyrandom.hpp` are a C++ port of
`kaggriculture.py` from `kaggle-environments` 1.32.7 (official source SHA256
`bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e`), published
under the Apache License 2.0 in the Kaggle kernel
[`yhay81/three-day-shop-router`](https://www.kaggle.com/code/yhay81/three-day-shop-router).

They are used here as a simulator only. No policy, tape, or guard from that
kernel is included; `bridge.cpp`, the C ABI our wrapper loads, is ours. The two
adopted files carry exactly one modification: a two-line comment at the top of
each naming the kernel and pointing at
`src/kaggriculture/campaign/engine/NOTICE`, which travels in the submission
archive beside the compiled library.

The upstream software is distributed under the Apache License, Version 2.0:

> Licensed under the Apache License, Version 2.0 (the "License"); you may not
> use this file except in compliance with the License. You may obtain a copy of
> the License at
>
> > http://www.apache.org/licenses/LICENSE-2.0
>
> Unless required by applicable law or agreed to in writing, software
> distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
> WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
> License for the specific language governing permissions and limitations under
> the License.
