<h1 align= 'center'>Image denoising using ISING MODEL and GIBBS SAMPLING</h1>

<p align='center'>
Mathis Matthieu, Benjamin Pipaud, Lucas Saban. MC course @Ensae
</p>

---

<p align='center'>
Ising Model denoising using hyperparameter estimation and MCMC techniques.
</p>
         


## Examples

With systematic scan on the left and randomized scan on the right:
<p align="center">
 <img src="https://user-images.githubusercontent.com/73651505/165507759-cb2c4536-662f-4eae-85f7-e6ffce956d67.gif" alt="example" width="200"/>

 <img src="https://user-images.githubusercontent.com/73651505/167620431-46334231-9c49-4e89-8b4e-f8c3b5768801.gif" alt="random_gif" width="200"/>
</p>


## ⚡️ On Boarding 

```shell
pip install -r requirements.txt
```

Then : 

```shell
main.py --findsigma True --alpha 0.0 #etc...
```

The arguments available are :

- `alpha` : The alpha parameter of the Ising Model, default value is 0
- `beta`: The beta parameter of the Ising Model, default value is 1.3
- `sigma` : Variance of the gaussian noise, default value is 179 (for 8 bit images)
- `findsigma` : If set to True, the denoising will be done without being given the value of sigma, default value False
- `g` : If set to true a gif will be produced. Default value to True.
- `b` : Number of burn in steps. Default to 40
- `ns`: Number of sampling steps. Default to 5
- `imp`: Path of the raw image. Default to 'data/input/test_img.jpeg'
