#include<bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool lemonadeChange(vector<int>& bills) {
        int five = 0, ten = 0;
        for(int i = 0 ; i < (int)bills.size() ; i++) {
            if(bills[i] == 5) {
                five++;
            } else if(bills[i] == 10) {
                if(five == 0) {
                    return false;
                }
                five--;
                ten++;
            } else {
                if(five > 0 && ten > 0) {
                    five--;
                    ten--;
                } else if(five >= 3) {
                    five -= 3;
                } else {
                    return false;
                }
            }
        }
        return true;
    }
};
int main() 
{
    Solution s;
    int n;
    cin >> n;
    vector<int> bills(n);
    for(int i=0; i<n; i++) {
        cin >> bills[i];
    }
    bool ans = s.lemonadeChange(bills);
    ans ? cout << "true" : cout << "false";
} 