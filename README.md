We propose the following improvements to the paper Wang, M., & Ku, H. (2022). *Risk-sensitive policies for portfolio management*. Expert Systems with Applications, 198, 116807. https://doi.org/10.1016/j.eswa.2022.116807
- We show that it is possible the change the return distribution to a non-standardized t-distribution, as long as the degrees of freedom is fixed, since in that case, t-distributions are closed under affine transformations. The convergence of critic network is preserved and computation-wise, it is the same as using a normal distribution.
- The paper claimed their proposed Distributional DDPG algorithm is able to adapt to different alpha values, by changing its policy to more agressive or conservative. However, their implementation only works for a fixed value of alpha and they trained different models for different values and evaluated those. We propose an improved version of this algorithm that is truly alpha-sensitive, so only a single model has to be trained and it can be used for any (reasonable) alpha level.
- We rigorously evaluated and benchmarked our models against the proposed algorithms in the paper and common baselines.
- Refactored the original code to make it easier to work with and made it more modular so that anyone wanting to contribute can iterate on their ideas more quickly.


# Setting up the project
All of the code is intended to be run in the same environment as the original, so we recommend following the instructions in https://codeocean.com/capsule/0769244/tree/v1 and using the Dockerfile to set up a container.

The main entry point is `src/`. It contains the files which can be used to reproduce our results. `video/` contains the pre-trained models, so it is not needed to retrain the models for reproduction. To reproduce the plots in `results/`, run the following code (after setting up the Docker environment):
```
docker run -w /code -v "$(pwd)/code:/code" -v "$(pwd)/data:/data" [name-of-container] python -u eval/compare.py
```
