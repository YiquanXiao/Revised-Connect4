Name: Yiquan Xiao
Name of bot: BetaGo
Description of evaluation function: I use score of the current board + 3 * number of open-ended 2-in-a-row for each player as the utility function when the game hasn't finished. First, since the score is what we care most, score of the current board must be one of the feature for the evaluation function. Then, I believe the number of open-ended 2-in-a-row is also a feature. The more open-ended 2-in-a-row you have, you are more likely to get a score of 3^2 (9) in your next step and the opponent are less likely to prevent you to get those points. Also, I decide to use 3 * (# of open-ended 2-in-a-row) because these are like the "potential points" for the game, not true score of the board. So for each open-ended 2-in-a-row, I can't let it larger than the true score 4. As a result, I choose 3 as the coefficient for this feature. 
In the test_boards.py, I made some boards which has the name that represent where the next player should do. 
Note: 
1. If we reach the end of the game before depth becomes 0, we shhould directly use score() instead of evaluation.
2. Using a helper function for alpha-beta pruning will make it much easier. 
