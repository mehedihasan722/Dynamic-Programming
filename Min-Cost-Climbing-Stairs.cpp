#include<bits/stdc++.h>
using namespace std;

class Solution {
public:
    int minCostClimbingStairs(vector<int>& cost) {
        int n = (int) cost.size();
        vector<int> dp(n+1);
        dp[0]=dp[1]=0;
        for(int i=2 ; i<=n ; i++) {
            int x = dp[i-1] + cost[i-1];
            int y = dp[i-2] + cost[i-2];
            dp[i] = min(x ,y );
        }
        return dp[n];
    }
};
int main() 
{
    Solution s;
    int n;
    cin >> n;
    vector<int> cost(n);
    for(int i = 0 ; i < n ; i++) {
        cin >> cost[i];
    }
    int ans = s.minCostClimbingStairs(cost);
    cout << ans << endl;
}