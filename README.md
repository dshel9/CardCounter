# CardReader
CardReader is a program that counts cards for the user. It is designed to be used for a game of Hearts played on [Trickster Cards](https://www.trickstercards.com/home/hearts/). 
## Description
The program captures the user's screen. Once four cards are detected as being played, the top left corners of the cards are passed to the the suit classifier and number classifier. The four played cards are crossed off the board. If any of the players don't follow the lead suit, meaning they are out of that suit, the suit is crossed off for the player. Once the scoreboard appears, meaning the round is over, the board resets for the next hand. 

### How to run
To run, first navigate to [Trickster Cards](https://www.trickstercards.com/home/hearts/) and join a game against the computer (Play now on this site -> Hearts -> Practice against the computer -> Hearts | 4 player). Then run LineUpScreen.py
```
python3 LineUpScreen.py
```
<img width="735" alt="LineUpScreenNoCardsPlayed" src="https://github.com/user-attachments/assets/dcae8601-d9a0-4300-bdc7-8bc887ea3560" />


This will display the window that's being captured with 4 blue rectangles, showing the areas that will be monitered for played cards. 
<img width="446" alt="LineUpScreenCardsPlayed" src="https://github.com/user-attachments/assets/3568cf4f-0f46-4257-9dcf-33d9f9862613" />

Play a few tricks and adjust your browser so the cards line up as in the photo. Keep in mind the cards aren't played in the same places everytime.

Once you're satisfied with how the cards line up click on the window showing the rectangles and press q to quit the program. Now it's time to run the main program. 
```
python3 main.py
```
<img width="977" alt="CountWindowNoCrosses" src="https://github.com/user-attachments/assets/9f6a33c0-350f-4950-b1d8-251a44f0602e" />
Once the count window appears and "READY TO PLAY" is printed to terminal you can pass your cards and begin play. Each time a trick is finished, you'll see the four played cards crossed off on the board as well as "Player i won the trick with X of Suit" where i is the player who won and X of Suit is the card they won it with. Anytime a player plays a different suit that what is lead, that suit is crossed off for that player, meaning they have no more of that suit left. 


<img width="977" alt="CountWindowManyCrosses" src="https://github.com/user-attachments/assets/628f7caa-677d-4300-946c-813672e678cd" />
Here is an example board near the end of a hand. There are only 8 cards that haven't been played, the 2, 4 and 5 of Diamonds, 5 and King of Hearts, 3 and 5 of Clubs and the 5 of Spades. You can tell this by seeing that they are the only 8 cards not yet crossed off. Player 0 has Diamonds crossed off. This is because during the course of play, that player played a non Diamond card when Diamonds were lead. This means Player 0 can't have Diamonds. Player 1 has Diamonds and Clubs crossed off and Player 3 has Clubs crossed off for the same reasoning. 

To finish running the main program, click on the board and press the space bar. 

Please do not have any other tabs open when running the program as this will affect the where the cards and scoreboard appear. 
