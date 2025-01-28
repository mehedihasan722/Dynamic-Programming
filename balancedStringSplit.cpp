#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int balancedStringSplit(string s) {
        int count = 0, ans = 0;
        for(int i=0; i<(int)s.size(); i++) {
            if(s[i] == 'R') {
                count++;
            } else {
                count--;
            }
            if(count == 0) {
                ans++;
            }
        }
        return ans;    
    }
};

int main() 
{
    Solution s;
    string str;
    cin >> str;
    int ans = s.balancedStringSplit(str);
    cout << ans;
}