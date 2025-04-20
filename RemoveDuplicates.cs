

public static class LeetChallenges
{


    // Works but inefficient
    public static int MajorityElement(int[] nums) {
            for (int i=0; i< nums.Length; i++){
                int counter = 0;            
                for (int j=0; i< nums.Length; j++){
                    if (nums[i] == nums[j]){
                        counter++;
                    }
                    if (counter >= nums.Length/2){
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