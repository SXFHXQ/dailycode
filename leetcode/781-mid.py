'''
森林中的兔子

森林中，每个兔子都有颜色。其中一些兔子（可能是全部）告诉你还有多少其他的兔子和自己有相同的颜色。
我们将这些回答放在 answers 数组里。返回森林中兔子的最少数量。

示例:
输入: answers = [1, 1, 2]
输出: 5

输入: answers = [10, 10, 10]
输出: 11

输入: answers = []
输出: 0
'''
class Solution:
    def numRabbits(self, answers: List[int]) -> int:
        if not answers:
            return 0
        ans=0
        d={}
        for key in answers:
            if key in d:
                d[key]+=1
            else:
                d[key]=1
        for key,value in d.items():
            if value%(key+1)==0:
                ans+=(value//(key+1))*(key+1)
            else:
                ans+=(value//(key+1)+1)*(key+1)
        return ans

#时间复杂度：O(N)
#空间复杂度：O(N)