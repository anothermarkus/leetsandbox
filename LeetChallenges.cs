
public class RandomizedSet
{

      public static int[] ProductExceptSelf(int[] nums) {
        int[] output = new int[nums.Length];

        //  [1,2,3,4]
        // Prefix - Basic Product Array - Multiply over each value over to the next one
        // Suffix - Product Array each element at index i contains the product of all elements to the right of i
        
        //left[i] = product of nums[0..i-1]
        //right[i] = product of nums[i+1..n-1]

        // [1,2,3,4] -> [1,  1,  2, 6]     // Prefix product (L)
        // [1,2,3,4] -> [24, 12, 4, 1]     // Suffix product (R)                                        
        // [24,12,8,6]                     // Result array

        // [1,2,3,4] x [1,2,3,4] 
        //      <-L           R   // nums[3]
        //    <-L           R->   // nums[2]
        //   <-L           R->    // nums[1]
        //  L           R->       // nums[0]
     
        int[] left  = new int[nums.Length];
        int[] right  = new int[nums.Length];

        left[0] = 1;
        right[nums.Length-1] = 1;

        for (int i = 1; i < nums.Length; i++) {
           //typically
           //nums[i] = nums[i] * nums[i - 1];
            left[i] = left[i - 1] * nums[i - 1];
        }

        for(int i=nums.Length-2;i>=0;i--)
        {
            right[i]=right[i+1]*nums[i+1];
            // typically an adjustment after to include 
            // a for-loop that includes i
            // right[i] *= right[i];
        }


        for(int i=0;i<nums.Length;i++)
        {
            output[i]=left[i]*right[i];
        }


        return output;
    }


    Dictionary<int, int> dictionary;
    List<int> list;
    Random random;

    public RandomizedSet()
    {
        dictionary = new Dictionary<int, int>();
        list = new List<int>();
        random = new Random();
    }

    public bool Insert(int val)
    {

        if (dictionary.ContainsKey(val))
        {
            return false;
        }

        list.Add(val);
        dictionary.Add(val, list.Count - 1);
        return true;
    }


    public bool Remove(int val)
    {
        if (!dictionary.ContainsKey(val))
        {
            return false;
        }

        var index = dictionary[val];
        var endValue = list[list.Count - 1];

        list[index] = endValue; //swap it out

        dictionary[endValue] = index;

        // remove the value in dictionary
        dictionary.Remove(val);

        // remove the value in the list
        list.RemoveAt(list.Count - 1);

        return true;
    }

    public int GetRandom()
    {
        int randomVal = random.Next(0, list.Count);
        return list[randomVal];
    }
}



public static class LeetChallenges
{





    // First we order the values of f from the largest to the lowest value. 
    // Then, we look for the last position in which f 
    // is greater than or equal to the position (we call h this position)
    // Sorting kills the time complexity
    // However! you can use storage to gain O(n) time at the cost of O(n) size
    public static int HIndex(int[] citations)
    {

        // [3,0,6,1,5]
        int n = citations.Length;

        if (n == 0) return 0;

        Array.Sort(citations);
        Array.Reverse(citations);

        //  sort and check from the end
        // [6,5,3,1,0] - f's
        //  1 2 3 4 5  - position

        // [3,1,1]
        //  1 2 3

        int i = citations.Length - 1;
        while (i >= 0)
        {
            if (citations[i] >= (i + 1))
            {
                return i + 1;
            }
            i--;
        }
        return 0;
    }


    public static int JumpII(int[] nums)
    {
        int jumps = 0;
        int currentEnd = 0;
        int farthest = 0;

        for (int i = 0; i < nums.Length - 1; i++)
        {
            farthest = Math.Max(farthest, i + nums[i]);

            // Even though you are not actually 
            // jumping from this point, "farthest"
            // will be tracking how long you can go
            // you already passed your jump point
            // and just collecting your longest distance 
            // as you go. 
            if (i == currentEnd)
            {
                jumps++;
                currentEnd = farthest;
            }
        }

        return jumps;
    }

    public static bool CanJump(int[] nums)
    {

        // (*) [* x]   [*]
        // [2, 3, 1, 1, 4, 2] - true
        int biggest = 0;

        for (int i = 0; i < nums.Length; i++)
        {

            // If I ever iterate beyond my max range
            // then the jump to the end is impossible return false
            if (i > biggest)
            {
                return false;
            }

            biggest = Math.Max(biggest, i + nums[i]);
        }
        return true;
    }


    public static int MaxProfitSpaceOptimized(int[] prices)
    {
        if (prices.Length == 0) return 0;

        int hold = -prices[0];
        int notHold = 0;

        for (int i = 1; i < prices.Length; i++)
        {
            int prevNotHold = notHold;
            notHold = Math.Max(notHold, hold + prices[i]);
            hold = Math.Max(hold, prevNotHold - prices[i]);
        }

        return notHold;
    }

    public static int MaxProfitDP(int[] prices)
    {
        if (prices.Length == 0) return 0;

        int n = prices.Length;
        int[,] dp = new int[n, 2]; // Memoize: store all computations in array

        // dp[i, 0] = max profit at day i when NOT holding
        // dp[i, 1] = max profit at day i when HOLDING


        dp[0, 0] = 0;             // Not holding on day 0: no stock, no profit
        dp[0, 1] = -prices[0];    // Holding on day 0: bought the stock, profit is negative cost

        // greedy approach
        for (int i = 1; i < n; i++)
        {
            // Not holding on day i:
            // Either we did nothing (stayed not holding), or we sold today (was holding yesterday)
            dp[i, 0] = Math.Max(dp[i - 1, 0], dp[i - 1, 1] + prices[i]);

            // Holding on day i:
            // Either we did nothing (kept holding), or we bought today (was not holding yesterday)
            dp[i, 1] = Math.Max(dp[i - 1, 1], dp[i - 1, 0] - prices[i]);
        }

        return dp[n - 1, 0]; // Final profit must be when not holding stock
    }


    // Linear Time
    public static int MaxProfit(int[] prices)
    {
        int maxProfit = 0;
        for (int i = 1; i < prices.Length; i++)
        {
            if (prices[i] > prices[i - 1])
            {
                maxProfit += prices[i] - prices[i - 1];
            }
        }
        return maxProfit;
    }



    // Brute Force recursion
    public static int MaxProfitMedium(int[] prices)
    {
        return BuySell(prices, 0);
    }

    private static int BuySell(int[] prices, int start)
    {
        if (start >= prices.Length)
            return 0;

        int maxProfit = 0;
        for (int buy = start; buy < prices.Length - 1; buy++)
        {
            for (int sell = buy + 1; sell < prices.Length; sell++)
            {
                if (prices[sell] > prices[buy])
                {
                    int currentProfit = prices[sell] - prices[buy];
                    // After selling, you can start buying again from sell + 1
                    int remainingProfit = BuySell(prices, sell + 1);
                    maxProfit = Math.Max(maxProfit, currentProfit + remainingProfit);
                }
            }
        }

        return maxProfit;
    }




    // don't have to iterate over every combination
    // because you are guaranteed to capture 
    // the most profit even if a lower min price
    // comes up later
    public static int MaxProfit2(int[] prices)
    {
        int minPrice = int.MaxValue;
        int maxProfit = 0;

        foreach (int price in prices)
        {
            if (price < minPrice)
            {
                minPrice = price;
            }
            else
            {
                int profit = price - minPrice;
                if (profit > maxProfit)
                {
                    maxProfit = profit;
                }
            }
        }

        return maxProfit;
    }


    public static int MaxProfit(int[] prices)
    {

        int profit = 0;

        for (int i = 0; i < prices.Length; i++)
        {
            for (int j = i + 1; j < prices.Length; j++)
            {
                // is this more profitable?
                if (prices[j] - prices[i] > profit)
                {
                    profit = prices[j] - prices[i];
                }
            }
        }

        return profit;

    }


    //  k = 3
    // [1, 2, 3, 4, 5, 6, 7] - Input Array
    // [7, 6, 5, 4, 3, 2, 1] - Reverse All
    // [5, 6, 7, 4, 3, 2, 1] - Reverse first k
    // [5, 6, 7, 1, 2, 3, 4] - Reverse remaining
    public static void Rotate2(int[] nums, int k)
    {
        int n = nums.Length;
        k %= n; // In case k is huge!
        Reverse(nums, 0, n - 1);
        Reverse(nums, 0, k - 1);
        Reverse(nums, k, n - 1);
    }

    private static void Reverse(int[] nums, int left, int right)
    {
        while (left < right)
        {
            int temp = nums[left];
            nums[left] = nums[right];
            nums[right] = temp;
            left++;
            right--;
        }
    }




    // Brute force
    // Tried to do something elegant with swapping values
    // but only worked when the array had an odd number of values
    public static void Rotate(int[] nums, int k)
    {

        for (int i = 0; i < k; i++)
        {

            int curIndex = 0;
            int currentValue = nums[0];
            int valueBeforeOverwrite;

            for (int j = 1; j < nums.Length; j++)
            {
                curIndex = (curIndex + 1) % nums.Length;

                valueBeforeOverwrite = nums[curIndex];
                nums[curIndex] = currentValue;
                currentValue = valueBeforeOverwrite;
            }
            nums[0] = currentValue;

        }


    }


    // add one count each time you find a match
    // subtract one count each time you don't
    // whatever has a net value of > 0 is the winner
    // we don't have to have a total count of the majority
    // all we need to do is to use a strategy that 
    // cancels whatever is currently the majority if it doesn't match
    // whatever remainder will still find the majority
    public static int BoyerMooreMajority(int[] nums)
    {
        int count = 0;
        int candidate = 0;

        foreach (int num in nums)
        {
            if (count == 0)
            {
                candidate = num;
            }
            count += (num == candidate) ? 1 : -1;
        }

        return candidate;
    }

    // Works but inefficient
    public static int MajorityElement(int[] nums)
    {
        for (int i = 0; i < nums.Length; i++)
        {
            int counter = 0;
            for (int j = 0; i < nums.Length; j++)
            {
                if (nums[i] == nums[j])
                {
                    counter++;
                }
                if (counter >= nums.Length / 2)
                {
                    return nums[i];
                }
            }
        }
        return nums[0];
    }



    // For simplicity if there are two numbers in the array just return
    // Duplicates are okay up to two, and any two numbers are fine

    // i - write pointer, only increments after a write (slow-runner )
    // j - read pointer, checks current value against two ago from i
    // if it's the same, it means we have more than one duplicate always moves forward

    // This method allows for at most one duplicate
    public static int RemoveDuplicates2(int[] nums)
    {

        if (nums.Length <= 2) return nums.Length;

        int i = 2; // Start from 3rd position
        for (int j = 2; j < nums.Length; j++)
        {
            if (nums[j] != nums[i - 2])
            {
                nums[i] = nums[j];
                i++;
            }
        }

        return i; // i is the new length
    }


    public static int RemoveDuplicates(int[] nums)
    {
        if (nums.Length == 0) return 0;

        int i = 0; // Pointer to the last unique element
        for (int j = 1; j < nums.Length; j++)
        {
            if (nums[j] != nums[i])
            {
                i++;
                nums[i] = nums[j]; // Place the next unique value
            }
        }

        return i + 1; // Number of unique elements
    }

    public static int RemoveElement(int[] nums, int val)
    {
        int i = 0;
        int n = nums.Length;

        while (i < n)
        {
            if (nums[i] == val)
            {
                nums[i] = nums[n - 1]; // Overwrite with last valid element
                n--; // Reduce effective array size
            }
            else
            {
                i++;
            }
        }

        return n; // New length
    }




    public static void Merge(int[] nums1, int m, int[] nums2, int n)
    {
        int writeIndex = m + n - 1;  // Last position in nums1
        int i = m - 1;  // Last initialized element in nums1
        int j = n - 1;  // Last element in nums2

        if (m == 0)
        {
            while (j >= 0)
            {
                nums1[writeIndex--] = nums2[j--];
            }
            return;
        }

        // Merge in reverse order
        while (i >= 0 && j >= 0)
        {
            if (nums1[i] > nums2[j])
            {
                nums1[writeIndex--] = nums1[i--];
            }
            else
            {
                nums1[writeIndex--] = nums2[j--];
            }
        }

        // If any elements are left in nums2, copy them
        while (j >= 0)
        {
            nums1[writeIndex--] = nums2[j--];
        }

        // No need to copy nums1's elements—they're already in place
    }




}