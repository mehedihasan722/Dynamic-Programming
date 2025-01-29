#include <bits/stdc++.h>
using namespace std;
/**
 * @file houseRobber.cpp
 * @brief This file contains the solution to the House Robber problem using dynamic programming.
 * 
 * The House Robber problem is a classic dynamic programming problem where the goal is to determine
 * the maximum amount of money that can be robbed from a series of houses, given that no two adjacent
 * houses can be robbed on the same night.
 * 
 * The solution involves creating a dynamic programming table to store the maximum amount of money
 * that can be robbed up to each house, and then using this table to compute the final result.
 * 
 * The time complexity of this solution is O(n), where n is the number of houses, and the space
 * complexity is also O(n) due to the use of the dynamic programming table.
 */

class Solution {
public:
    /**
     * @brief Computes the maximum amount of money that can be robbed without robbing two adjacent houses.
     * 
     * This function uses dynamic programming to determine the maximum amount of money that can be robbed
     * from a list of houses, where each house has a certain amount of money. The constraint is that no two
     * adjacent houses can be robbed on the same night.
     * 
     * @param nums A vector of integers representing the amount of money in each house.
     * @return int The maximum amount of money that can be robbed.
     */
    /**
     * @brief This function solves the House Robber problem using dynamic programming.
     * 
     * The House Robber problem is a classic dynamic programming problem where a robber
     * needs to maximize the amount of money he can rob from a series of houses. However,
     * he cannot rob two adjacent houses because the security system will alert the police.
     * 
     * @param nums A vector of integers representing the amount of money in each house.
     * @return int The maximum amount of money that can be robbed without alerting the police.
     * 
     * The function follows these steps:
     * 1. If there are no houses, return 0.
     * 2. If there is only one house, return the amount of money in that house.
     * 3. If there are two houses, return the maximum amount of money between the two houses.
     * 4. Create a dp vector to store the maximum amount of money that can be robbed up to each house.
     * 5. Initialize the first two elements of the dp vector.
     * 6. Iterate through the houses starting from the third house, updating the dp vector with the maximum
     *    amount of money that can be robbed up to that house.
     * 7. Return the last element of the dp vector, which contains the maximum amount of money that can be robbed.
     */
    int rob(vector<int>& nums) {
        int n = nums.size();
        if(n == 0) {
            return 0;
        }
        if(n == 1) {
            return nums[0];
        }
        if(n == 2) {
            return max(nums[0], nums[1]);
        }
        vector<int> dp(n, 0);
        dp[0] = nums[0];
        dp[1] = max(nums[0], nums[1]);
        for(int i=2; i<n; i++) {
            dp[i] = max(dp[i-1], dp[i-2] + nums[i]);
        }
        return dp[n-1];
    }
};

int main() 
{
    Solution s;
    vector<int> nums;
    int n, num;
    cin >> n;
    for(int i=0; i<n; i++) {
        cin >> num;
        nums.push_back(num);
    }
    int ans = s.rob(nums);
    cout << ans;
}