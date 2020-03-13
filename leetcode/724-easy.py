'''
寻找数组的中心索引
给定一个整数类型的数组 nums，请编写一个能够返回数组“中心索引”的方法。
我们是这样定义数组中心索引的：数组中心索引的左侧所有元素相加的和等于右侧所有元素相加的和。
如果数组不存在中心索引，那么我们应该返回 -1。如果数组有多个中心索引，那么我们应该返回最靠近左边的那一个。

示例 1:
输入: 
nums = [1, 7, 3, 6, 5, 6]
输出: 3

示例 2:
输入: 
nums = [1, 2, 3]
输出: -1
'''
class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        if not nums:return -1
        if len(nums)==1 or sum(nums[1:])==0 :return 0

        h=sum(nums[1:])
        q=0
        for i in range(1,len(nums)-1):
            q+=nums[i-1]
            h-=nums[i]
            if q==h:
                return i

        if sum(nums[:-1])==0:return len(nums)-1
        
        return -1

#时间复杂度O(N)
#空间复杂度O(1)

class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        S=sum(nums)
        left=0
        for i,value in enumerate(nums):
            if left==S-left-nums[i]:
                return i
            left+=nums[i]
        return -1
#时间复杂度O(N)
#空间复杂度O(1)