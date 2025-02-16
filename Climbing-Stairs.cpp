#include<bits/stdc++.h>
using namespace std;

class Solution {
public:
    int climbStairs(int n) {
        vector<int> ways(n+1);
        ways[0]=1,ways[1]=1;
        for(int i = 2 ; i <= n ; i++) {
            ways[i] = ways[i-1] + ways[ i - 2];
        }
        return ways[n];
    }
};

int main() 
{
    Solution s;
    int n;
    cin >> n;
    int ans = s.climbStairs(n);
    cout << ans << endl;
}

