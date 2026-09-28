# Third-party code, models and images

## llm-robotics-playground

The drawing and writing scenes run inside [llm-robotics-playground](https://github.com/dimentary/llm-robotics-playground)
by Dmitry Hrybov, which is not included here. The servers in `scenes/` are built on its dove-drawing and
fibonacci-writing experiments: they import that project's controller, firmware and writer at runtime from a checkout
you clone yourself. Its original code is under the MIT License:

```
MIT License

Copyright (c) 2026 Dmitry Hrybov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The playground's dove artwork and the assets derived from it are excluded from its MIT grant. statebench does not
include or use them: every drawing here uses the targets and references listed below.

## Robot models

None are included. The xArm7 (pick and place, button) comes from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (`ufactory_xarm7`, BSD-3-Clause, UFACTORY).
The Kinova Gen3, Shadow Hand and Unitree G1 used by the drawing and writing scenes come with llm-robotics-playground,
which takes them from MuJoCo Menagerie, and keep their upstream licenses.

## Images in `assets/`

All are public domain, from Wikimedia Commons.

| File | Work | Source file | Why public domain |
|---|---|---|---|
| `paintings/mona_lisa.jpg` | Mona Lisa, Leonardo da Vinci, about 1503-1519 | `Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg` | artist died 1519 |
| `paintings/starry_night.jpg` | The Starry Night, Vincent van Gogh, 1889 | `Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg` | artist died 1890 |
| `paintings/great_wave.jpg` | The Great Wave off Kanagawa, Katsushika Hokusai, about 1831 | `Tsunami_by_hokusai_19th_century.jpg` | artist died 1849 |
| `references/cat.png` | Cat silhouette | [`Cat_silhouette.svg`](https://commons.wikimedia.org/wiki/File:Cat_silhouette.svg) | released by its author (PD-self) |
| `references/fish.png` | Fish icon | [`Fish_icon.svg`](https://commons.wikimedia.org/wiki/File:Fish_icon.svg) | released by its author, the Swedish Road Administration (PD-author) |
| `references/star.png` | Five pointed star | [`Five_Pointed_Star_Solid.svg`](https://commons.wikimedia.org/wiki/File:Five_Pointed_Star_Solid.svg) | simple geometry (PD-shape) |

The paintings are photographs of 2D public-domain works, which Wikimedia Commons marks as public domain. The
`*_trace.json` files are outlines traced from these images by `scenes/trace_paintings.py`.

## Code from robolabel

`src/statebench/jsontext.py` is copied from [robolabel](https://github.com/kevdozer1/robolabel) (Apache-2.0, same
author).
