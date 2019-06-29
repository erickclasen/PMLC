# PMLC - Python Machine Learning Course
This is the code for a small course that introduces machine learning from the very basics.
The code accompanies and extends the blog posts that are stored at ...
http://erick.heart-centered-living.org/pmlc-2/

## Overview

The objective with this code is to show the very basics of machine learning by sneaking up on it using very simple math
and logic gate examples and then building on top of that. The course assumes some basic Python coding knowledge, basic
high school math and pretty much that is it. 

In some of the folders there are notes.txt files that give details about the code. This is a work in progress and some
of the code may not work, when possible I have made a note on issues.

## Outline


The outline will follow this ordering:

    Solvers - Basic Math Via Algorithm
        Brute Force
        Step Up and Jump Down
        Split Search
        Linear Proportional Solver - backpropagation
        Differential Evolution - to show a different way of doing it
    Perceptrons - Logic Gate Examples
        AND to XOR
        Solving XOR with 2 layers and non-linearity
        Solving XOR with  feature engineering
    Self Learning Perceptron, Fusion of Perceptron and Solver.
        Gates revisited using self learning via backpropigation
    Solver to Linear Regression
        Single Variable
        Dual Variable
        General Multivariate
    Feedforward Networks
        Decimal to Binary 10 in 4 out Single Feed Forward Network
        Image Recognition - Simplified Digit Recognition

##Directories so far
.
├── activations *
├── classifiers 
├── gates
├── layered-n-n
├── lin-reg-basic
├── numpy-warmup-predict-y-from-x *
├── perceptrons
├── solvers
│   └── split-search
├── Vectors_Linear_Algebra **
└── xor

* May be incomplete
** Bonus material




## History

I came up with the idea to create this after getting stumped trying to dive in and use machine learning. Yes it is
possible to use it like an "appliance" by using tools like Tensorflow. 

You can do tons of cool stuff with it all day long by treating it like an appliance. 
But, in my opinion it is interesting and worth examining how it actually works by starting from the very basics.

After getting blown away by some of the resources I initially came across,
I decided to ask the question. What would the absolute most basic example of machine learning look like? I took the 
viewpoint of what if we could go back in time before powerful computers and think of how it would be to invent it from
scratch. So I started working those examples and even went a little backwards to sneak up on the problem from a viewpoint
of "beginners mind". I quickly found out that by getting back to basics, learning accelerated. At first it seemed kind
of silly to work right down to the very basics but, it actually was worth it.

Since this approach worked for me, I have decided to post this code here so that others may benefit from it.

## Dependencies
Some of the examples require numpy. I have tried to keep the requirements as simple as possible. This is a work
in progress and if I missed a dependancy I will add it in TBD.

* numpy for some examples

Install missing dependencies with [pip](https://pip.pypa.io/en/stable/)

## Usage

Run `python name_of_file.py` in terminal and it will run.

## Credits

See the resources.txt file for the resources that this code has been built upon.
