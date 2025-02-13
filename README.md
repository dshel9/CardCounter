# CardReader
CardReader is a program that counts cards for the user. It is designed to be used for a game of Hearts played on [Trickster Cards](https://www.trickstercards.com/home/hearts/). 
## Description
The program captures the user's screen. Once four cards are detected as being played, the top left corners of the cards are passed to the the suit classifier and number classifier. The four played cards are crossed off the board. If any of the players don't follow the lead suit, meaning they are out of that suit, the suit is crossed off for the player. Once the scoreboard appears, meaning the round is over, the board resets for the next hand. 

### How to run
To run, first navigate to [Trickster Cards](https://www.trickstercards.com/home/hearts/) and join a game against the computer (Play now on this site -> Hearts -> Practice against the computer -> Hearts | 4 player). Then run LineUpScreen.py
```
python3 LineUpScreen.py
```
<img width="852" alt="LineUpScreenNoCards" src="https://github.com/user-attachments/assets/81dab4e4-5aef-4c22-a66d-7cf04eabf8f9" />



This will display the window that's being captured with 4 blue rectangles and 3 red rectangles, showing the areas that will be monitered for played cards and the scoreboard. 
<img width="808" alt="LineUpScreen4Cards1" src="https://github.com/user-attachments/assets/006bafa6-e3d3-418f-814e-fce75eb63ed1" />
<img width="808" alt="LineUpScreen4Cards2" src="https://github.com/user-attachments/assets/5376bd13-5fa1-4b54-b4a7-9bd92a2d4048" />

Play a few tricks and adjust your browser so the cards line up with the blue rectangles as in the photos. Keep in mind the cards aren't played in the same places everytime.

<img width="808" alt="LineUpScreenScoreboard" src="https://github.com/user-attachments/assets/215558b2-cdf3-4728-8ca4-43bc23071b4d" />

Once the hand is over, the scoreboard will appear. The red rectangles should align similar to the photo, with in between the colomns. 



Once you're satisfied with how the cards line up click on the window showing the rectangles and press q to quit the program. Now it's time to run the main program. 
```
python3 main.py
```
<img width="977" alt="CountWindowNoCrosses" src="https://github.com/user-attachments/assets/9f6a33c0-350f-4950-b1d8-251a44f0602e" />
Once the count window appears and "READY TO PLAY" is printed to terminal you can pass your cards and begin play. Each time a trick is finished, you'll see the four played cards crossed off on the board as well as "Player i won the trick with X of Suit" where i is the player who won and X of Suit is the card they won it with. Anytime a player plays a different suit that what is lead, that suit is crossed off for that player, meaning they have no more of that suit left. 


<img width="977" alt="CountWindowManyCrosses" src="https://github.com/user-attachments/assets/628f7caa-677d-4300-946c-813672e678cd" />
Here is an example board near the end of a hand. There are only 8 cards that haven't been played, the 2, 4 and 5 of Diamonds, 5 and King of Hearts, 3 and 5 of Clubs and the 5 of Spades. You can tell this by seeing that they are the only 8 cards not yet crossed off. Player 0 has Diamonds crossed off. This is because during the course of play, that player played a non Diamond card when Diamonds were lead. This means Player 0 can't have Diamonds. Player 1 has Diamonds and Clubs crossed off and Player 3 has Clubs crossed off for the same reasoning. 

To finish running the main program, click on the board and press the space bar. 

## File Breakdown
LabeledSuits/trainingDataX.npy - contains the training images as (35,25) 2D arrays. Scaled 0-1

LabeledSuits/trainingDataY.npy - contains the suit classifications as one hot arrays for all the images in LabeledSuits/trainingDataX.npy

LabeledNumbers/TrainindDataY.npy - contains the number classifications as one hot arrays for all the images in LabeledSuits/trainingDataX.npy

PredictedNumbers/X.npy - contains images the number model was run on after initial training

PredictedNumbers/X.npy - contains the predictions of the number model

PredictedSuits/Imgs.npy - contains images the suits model was run on after initial training

PredictedSuits/predictions.npy - contains the predictions of the suits model

TestIMG/TestImage1.png - screenshot of the window when all four cards were played, used to find the correct areas to give to the model

TestIMG/TestImage2.png, TestIMG/TestImage3.png, TestIMG/TestImage4.png, TestIMG/TestImage5.png - screenshots of the scoreboard of a euchre game, used to learn how the scoreboard changes after multiple rounds of play

TestIMG/TestImage6.png - screenshot of the scoreboard of a hearts game, used to find the correct areas to moniter to find if the scoreboard is showing

Game.py - class for counting the cards 

Helper.py - file containing various functions used during developement

ImageModifier.py - class for taking the screenshot and returning 4 numpy arrays, one of each region of interest for each card

find_region.py - file containing functions used to find the regions of the cards that contain the suit and number

main.py - runs the card counting program

LineUpScreen.py - runs the program to show you the window being captured and the regions the program is looking at for detecting cards and the scoreboard

model.py - creates the machine learning model for classifying suits and numbers 

numberClassifierV2.keras - keras model to classify the number of a card, input shape (35,25,1)

suitClassifierv2.keras - keras model to classify the suit of a card, input shape (35,25,1)
