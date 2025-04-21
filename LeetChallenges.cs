

public static class LeetChallenges
{


  // don't have to iterate over every combination
  // because you are guaranteed to capture 
  // the most profit even if a lower min price
  // comes up later
  public static int MaxProfit2(int[] prices) {
        int minPrice = int.MaxValue;
        int maxProfit = 0;

        foreach (int price in prices) {
            if (price < minPrice) {
                minPrice = price;
            } else {
                int profit = price - minPrice;
                if (profit > maxProfit) {
                    maxProfit = profit;
                }
            }
        }

        return maxProfit;
    }


    public static int MaxProfit(int[] prices) {

        int profit = 0;

        for (int i=0; i< prices.Length; i++){
            for (int j=i+1; j < prices.Length; j++){
                // is this more profitable?
                if (prices[j] - prices[i] > profit){
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