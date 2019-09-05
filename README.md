# FIFA Data Analysis Project

Data analysis project engaged with official FIFA 18 dataset (via kaggle).

The project goals are:

* Prediction of players FIFA scores, while minimizing the deviation comparing to the ground-truth scores.

* Creation of formations and tactics for a given club-squad, and finding the ideal players orders for each formation.

## Getting Started (for Windows or Linux)

### Installing

Running these commands will get you a copy of the project on your local machine, as well as install all relevant libraries:

```
git clone https://github.com/omermadmon/prj_FIFA.git
cd prj_FIFA
pip install -r requirements.txt
```

### Running

#### * Predicting FIFA scores and creating tactics for a certain club

```
python main.py juve
```

Note: 'juve' is an argument standing for the club name.

It can also be: barca, real madrid, liverpool, etc.

Classifier will be set to ORIENTED mode, taking into consideration players strong/weak foot while predicting score per position. 
It might increase the average deviation per position (FIFA does not take this criterion into account).

Results and loggings will be written to text files at the directories 'Results' and 'Logs'.


#### * Cross-validation

```
python main.py cv
```

Performing five-folds cross-validation.

Classifier will be set to FIFA mode, tracing FIFA score algorithm.

Loggings will be written to text files at 'Logs' directory, and cross-validation result will be written to the console.



*Running the program without an argument or giving a non-exsisting club name (such as 'Hapoel Haifa') will throw an exception.*


## Authors

* **Omer Madmon** 

* **Omer Nahum** 
