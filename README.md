# FIFA Data Analysis Project

Data analysis project engaged with official FIFA 18 dataset (via kaggle).

The project goals are:

* Prediction of players FIFA scores, while minimizing the deviation comparing to the ground-truth scores.

* Creation of formations and tactics for a given club-squad, and finding the ideal players orders for each formation.

## Getting Started (for Windows or Linux)

Running these commands will get you a copy of the project up and running on your local machine:

```
git clone https://github.com/omermadmon/prj_FIFA.git
cd prj_FIFA
pip install -r requirements.txt
python main.py juve
```

Note: 'juve' is an argument standing for the club name for predicting scores and creating tactics.

It can also be: barca, real madrid, liverpool and more.

Running the program without an argument or giving a non-exsisting club name (such as 'Hapoel Haifa') will throw an exception.

Results and loggings will be written to text files at the directories 'Results' and 'Logs'.


## Authors

* **Omer Madmon** 

* **Omer Nahum** 
