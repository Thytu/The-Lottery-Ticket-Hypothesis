<div id="top"></div>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


<br />
<div align="center">
  <a href="https://github.com/Thytu/The-Lottery-Ticket-Hypothesis">
    <img src="https://img.icons8.com/external-justicon-flat-justicon/344/external-lottery-gambling-justicon-flat-justicon.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">The Lottery Ticket Hypothesis</h3>

  <p align="center">
    Implementation of the "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks" paper.
    <br />
    <a href="#usage"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="#usage">View results</a>
    · <a href="https://github.com/Thytu/The-Lottery-Ticket-Hypothesis/issues">Report Bug</a>
    · <a href="https://github.com/Thytu/The-Lottery-Ticket-Hypothesis/issues">Request Feature</a>
  </p>
</div>

<br/>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#videos">Videos</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<br/>


## About The Project

<br>The Lottery Ticket Hypothesis:<br/> A randomly-initialized, dense neural network contains a subnetwork that is initialized such that—when trained in isolation—it can match the test accuracy of the original network after training for at most the same number of iterations

I found the Lottery Ticket hypothesis fascinating so I decided to re-implement the paper (fully for fun).

Key features:
* Code and results with the `Lenet-300-100` architecture on MNIST dataset
* Code and results with the `Conv-2` architecture, variants of VGG (Simonyan & Zisserman, 2014) on CIFAR10 dataset
* Code and results with the `Conv-4` architecture, variants of VGG (Simonyan & Zisserman, 2014) on CIFAR10 dataset
* Code and results with the `Conv-6` architecture, variants of VGG (Simonyan & Zisserman, 2014) on CIFAR10 dataset


The paper also experiments with `Resnet-18` and `VGG-19` which I didn't had time to on include (yet).\
If you would like to add any of those models, please consider to fork this repo and to create a pull request.

<br/>

### Videos

<br/>

<div align="center">

  <h3 align="center">The Lottery Ticket Hypothesis - Paper discussion</h3>

  <br/>

  [![The Lottery Ticket Hypothesis](https://img.youtube.com/vi/aQ9r4kpWPv0/0.jpg)](https://youtu.be/aQ9r4kpWPv0)
  
  <br/>
  
  <h3 align="center">The Lottery Ticket Hypothesis - Live coding</h3>
  
  [![The Lottery Ticket Hypothesis](https://img.youtube.com/vi/Q5wels_MuIk/0.jpg)](https://youtu.be/Q5wels_MuIk)

</div>

<p align="right">(<a href="#top">back to top</a>)</p>

<br/>

### Built With

* [PyTorch](https://pytorch.org)
* [Matplotlib](https://matplotlib.org/)

<p align="right">(<a href="#top">back to top</a>)</p>

## Getting Started

To get a local copy up and running follow these simple example steps.

Make sure to install the python dependencies : `python3 -m pip install requirements.txt`
(having access to a GPU will greatly increase the training speed but it's not mandatory)

<p align="right">(<a href="#top">back to top</a>)</p>


## Usage

Each folder corresponds to one of the main experiments described in the paper:
* [Lenet-300-100 on MNIST](./Lenet-300-100/README.md)
* [Conv-2 on CIFAR10](./Conv2/README.md)
* [Conv-4 on CIFAR10](./Conv2/README.md)
* [Conv-6 on CIFAR10](./Conv2/README.md)

To reproduce the experiments, simply follow the insctructions described in each `README.md` file.

## Roadmap

- [X] Add a results section for each model architecture
- [X] Plot the evolution of iteration for early-stopping
- [ ] Plot the evolution of iteration for early-stopping with weight resetting
- [X] Plot the graph based on the mean of five exeperiments
- [X] Add the min and max values in each plots
- [X] Add experiments with `Conv-2` on CIFAR10
- [X] Add experiments with `Conv-4` on CIFAR10
- [X] Add experiments with `Conv-6` on CIFAR10
- [ ] Add experiments with `Resnet-18` on CIFAR10
- [X] Add experiments with `VGG-19` on CIFAR10

See the [open issues](https://github.com/Thytu/The-Lottery-Ticket-Hypothesis/issues) for a full list of proposed features and known issues.

<p align="right">(<a href="#top">back to top</a>)</p>


## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make would improve this project, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!


1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/my-feature`)
3. Commit your Changes (`git commit -m 'feat: my new feature`)
4. Push to the Branch (`git push`)
5. Open a Pull Request

Please try to follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).

<p align="right">(<a href="#top">back to top</a>)</p>



## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>


## Contact

Valentin De Matos - [@ThytuVDM](https://twitter.com/ThytuVDM) - vltn.dematos@gmail.com

Project Link: [https://github.com/Thytu/The-Lottery-Ticket-Hypothesis](https://github.com/Thytu/The-Lottery-Ticket-Hypothesis)

<p align="right">(<a href="#top">back to top</a>)</p>


## Acknowledgments

* [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635)
* [Pytorch - Pruning](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
* [README Template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/Thytu/The-Lottery-Ticket-Hypothesis.svg?style=for-the-badge
[contributors-url]: https://github.com/Thytu/The-Lottery-Ticket-Hypothesis/graphs/contributors
[issues]: https://img.shields.io/github/issues/Thytu/The-Lottery-Ticket-Hypothesis
[forks-shield]: https://img.shields.io/github/forks/Thytu/The-Lottery-Ticket-Hypothesis.svg?style=for-the-badge
[forks-url]: https://github.com/Thytu/The-Lottery-Ticket-Hypothesis/network/members
[stars-shield]: https://img.shields.io/github/stars/Thytu/The-Lottery-Ticket-Hypothesis.svg?style=for-the-badge
[stars-url]: https://github.com/Thytu/The-Lottery-Ticket-Hypothesis/stargazers
[issues-shield]: https://img.shields.io/github/issues/Thytu/The-Lottery-Ticket-Hypothesis.svg?style=for-the-badge
[issues-url]: https://github.com/Thytu/The-Lottery-Ticket-Hypothesis/issues
[license-shield]: https://img.shields.io/github/license/Thytu/The-Lottery-Ticket-Hypothesis.svg?style=for-the-badge
[license-url]: https://github.com/Thytu/The-Lottery-Ticket-Hypothesis/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/valentin-de-matos
[product-screenshot]: .img/demo-simple.gif
