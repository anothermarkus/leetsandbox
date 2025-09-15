using System.Text;
using System.Collections;
using System.Net;

public class TreeNode {
      public int val;
      public TreeNode left;
      public TreeNode right;
      public TreeNode(int val=0, TreeNode left=null, TreeNode right=null) {
          this.val = val;
          this.left = left;
          this.right = right;
      }
  }


public class ListNode
{
    public int val;
    public ListNode next;

    public ListNode()
    {

    }
    public ListNode(int x)
    {
        val = x;
        next = null;
    }

    public ListNode(int x, ListNode next)
    {
        val = x;
        this.next = next;
    }

}

public class Node
{
    public int val;
    public Node next;
    public Node random;

    public Node(int _val)
    {
        val = _val;
        next = null;
        random = null;
    }
}

public class NodeConnect
{
    public int val;
    public NodeConnect left;
    public NodeConnect right;
    public NodeConnect next;

    public NodeConnect() { }

    public NodeConnect(int _val)
    {
        val = _val;
    }

    public NodeConnect(int _val, NodeConnect _left, NodeConnect _right, NodeConnect _next)
    {
        val = _val;
        left = _left;
        right = _right;
        next = _next;
    }
}

public class BSTIterator
{

    private Stack<TreeNode> stack;
    private TreeNode current;

    public BSTIterator(TreeNode root)
    {
        stack = new Stack<TreeNode>();
        current = root;
    }

    // In Order: LNR
    public int Next()
    {

        // L : Push all the lefts as it works recursively
        while (current != null)
        {
            stack.Push(current);
            current = current.left;
        }

        // N : Done with all the lefts, pop one out
        current = stack.Pop();
        int result = current.val;

        // R : Move right only once after lefts have been pushed first
        current = current.right;

        return result;
    }

    public bool HasNext()
    {
        return stack.Count > 0 || current != null;
    }
}


public class GraphNode
{
    public int val;
    public IList<GraphNode> neighbors;

    public GraphNode()
    {
        val = 0;
        neighbors = new List<GraphNode>();
    }

    public GraphNode(int _val)
    {
        val = _val;
        neighbors = new List<GraphNode>();
    }

    public GraphNode(int _val, List<GraphNode> _neighbors)
    {
        val = _val;
        neighbors = _neighbors;
    }
}

  public class Edge{
        public string Variable { get; set; } 
        public double Weight { get; set; }
        public Edge Next { get; set; }
    }

public class QNode
{
    public bool val;
    public bool isLeaf;
    public QNode topLeft;
    public QNode topRight;
    public QNode bottomLeft;
    public QNode bottomRight;

    public QNode()
    {
        val = false;
        isLeaf = false;
        topLeft = null;
        topRight = null;
        bottomLeft = null;
        bottomRight = null;
    }

    public QNode(bool _val, bool _isLeaf)
    {
        val = _val;
        isLeaf = _isLeaf;
        topLeft = null;
        topRight = null;
        bottomLeft = null;
        bottomRight = null;
    }

    public QNode(bool _val, bool _isLeaf, QNode _topLeft, QNode _topRight, QNode _bottomLeft, QNode _bottomRight)
    {
        val = _val;
        isLeaf = _isLeaf;
        topLeft = _topLeft;
        topRight = _topRight;
        bottomLeft = _bottomLeft;
        bottomRight = _bottomRight;
    }

}

public class MedianFinder {

    // Right Side
    PriorityQueue<int,int> minHeap = new PriorityQueue<int,int>();

    // Left Side
    PriorityQueue<int,int> maxHeap = new PriorityQueue<int,int>();

    public void AddNum(int num) {
        if (minHeap.Count == 0 || num >= minHeap.Peek()) {
            minHeap.Enqueue(num, num);
        } else {
            maxHeap.Enqueue(num, -num);
        }
        Rebalance();
    }
    
    public double FindMedian() {
        if (minHeap.Count == maxHeap.Count){
            return (minHeap.Peek() + maxHeap.Peek())/2.0;
        }

        if (minHeap.Count > maxHeap.Count){
            return minHeap.Peek();
        }
        return maxHeap.Peek();
    }

    private void Rebalance(){
        // check which side has more and move over
        if (minHeap.Count > maxHeap.Count + 1){
            int num = minHeap.Dequeue();
            maxHeap.Enqueue(num,-num);
        } else if (maxHeap.Count > minHeap.Count + 1){
            int num = maxHeap.Dequeue();
            minHeap.Enqueue(num,num);
        }
    }
}


static class LeetChallenges
{

     public static int MinDistance(string word1, string word2) {
        
        // dp[i][j] = minimum edits to convert the first  i characters of word1 
        //            into the first j characters of word2

        // Base - if last chars match
        // dp[i][j] = dp[i-1][j-1]

        // otherwise
        // Recurrence
        //dp[i][j] = min(
        // dp[i-1][j-1] + (word1[i-1] == word2[j-1] ? 0 : 1), //replace
        // dp[i-1][j] + 1,                                    //delete
        // dp[i][j-1] + 1                                     //insert
        int m = word1.Length;
        int n = word2.Length;

        // memo[i,j] = min edits for word1[0..i) vs word2[0..j)
        var memo = new int?[m + 1, n + 1];

        return FindMinDistance(m, n, word1, word2, memo);
    }

      private static int FindMinDistance(int i, int j, string word1, string word2, int?[,] memo) {
        // Base cases
        if (i == 0) return j; // need j inserts
        if (j == 0) return i; // need i deletes

        if (memo[i, j].HasValue) return memo[i, j].Value;

        if (word1[i - 1] == word2[j - 1]) {
            // chars match, no operation
            memo[i, j] = FindMinDistance(i - 1, j - 1, word1, word2, memo);
        } else {
            int replace = FindMinDistance(i - 1, j - 1, word1, word2, memo) + 1;
            int delete  = FindMinDistance(i - 1, j, word1, word2, memo) + 1;
            int insert  = FindMinDistance(i, j - 1, word1, word2, memo) + 1;

            memo[i, j] = Math.Min(replace, Math.Min(delete, insert)); // min of 3
        }

        return memo[i, j].Value;
    }

     public static bool IsInterleave(string s1, string s2, string s3) {

    // memo[s1pointer, s2pointer] =  
    // (s1[s1pointer]  == s3[s1pointer+s2pointer] => recurse( s1pointer++ )  ) || 
    //  (s2[s2pointer] == s3[s1pointer+s2pointer] ==>  recurse( s2pointer++ )  )

        if (s1.Length + s2.Length != s3.Length){ return false; }
        bool?[,] memo = new bool?[s1.Length + 1, s2.Length + 1];
        return CheckInterleave(0, 0, s1, s2, s3, memo);
    }

    private static bool CheckInterleave(int i, int j, string s1, string s2, string s3, bool?[,] memo) {
        if (i == s1.Length && j == s2.Length) return true; // boundary

        if (memo[i, j].HasValue) { return memo[i, j].Value; } // gotcha exit recusion

        int k = i + j; // index into s3
        bool ans = false;

        if (i < s1.Length && s1[i] == s3[k]) { // use up s1
            ans = ans || CheckInterleave(i + 1, j, s1, s2, s3, memo);
        }
        if (j < s2.Length && s2[j] == s3[k]) { // use up s2
            ans = ans || CheckInterleave(i, j + 1, s1, s2, s3, memo);
        }

        memo[i, j] = ans;
        return ans;
    }

    
     private static int start = 0, maxLen = 1;

    public static string LongestPalindrome(string s) {
        int n = s.Length;
        if (n < 2) return s;
        bool?[,] memo = new bool?[n, n];
        // two pointers, sweep all combinations [i/j ... -->...n]
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                if (IsPalindrome(s, memo,i, j) && j - i + 1 > maxLen) { 
                    start = i; // track start position
                    maxLen = j - i + 1; // new max
                }
            }
        }
        return s.Substring(start, maxLen);
    }

    private static bool IsPalindrome(string s, bool?[,] memo, int i, int j) {
        if (i >= j) return true; // base case, single char is palindrome
        if (memo[i, j] != null) return memo[i, j].Value; // Gotcha Cut down on recursions
        memo[i, j] = s[i] == s[j] && IsPalindrome(s, memo, i + 1, j - 1); //recurrence: if chars == and inner value is palindrome
        return memo[i, j].Value;
    }

    
    // current = down + right // recurrence: AllPaths[i][j] = AllPaths[i+1][j] + AllPaths[i][j+1]
    public static int UniquePathsWithObstacles(int[][] obstacleGrid)
    {
        int rows = obstacleGrid.Length;
        int cols = obstacleGrid[0].Length;
        int[][] memo = new int[rows][];
        for (int i = 0; i < rows; i++)
        {
            memo[i] = new int[cols];
            for (int j = 0; j < cols; j++)
            {
                memo[i][j] = -1;
            }
        }
        return AllPaths(obstacleGrid, memo, 0, 0);
    }

    private static int AllPaths(int[][] obstacleGrid, int[][] memo, int row, int col) {
        int rows = obstacleGrid.Length;
        int cols = obstacleGrid[0].Length;

        if (row >= rows || col >= cols){  return 0; } // Out of bounds        
        if (obstacleGrid[row][col] == 1){ return 0; } // Obstacle       
        if (row == rows - 1 && col == cols - 1){ return 1; } // Base Case       
        if (memo[row][col] != -1){ return memo[row][col]; } // Gotcha, cut down recursive paths
        
        int paths = AllPaths(obstacleGrid, memo, row + 1, col) 
                  + AllPaths(obstacleGrid, memo, row, col + 1); // Recurrence Relation
        memo[row][col] = paths;
        return paths;
    }


     // can only move down or right
    // memo[i][j] = min( memo[i+1][j], memo[i][j+1]  )
    public static int MinPathSum(int[][] grid)
    {
        int rows = grid.Length;
        int cols = grid[0].Length;

        int[][] memo = new int[rows][];
        for (int i = 0; i < rows; i++)
        {
            memo[i] = new int[cols];
            for (int j = 0; j < cols; j++)
            {
                memo[i][j] = -1;
            }
        }
        return MinSum(grid, memo, 0, 0);
    }

    private static int MinSum(int[][] grid, int[][] memo, int row, int col) {
        int rows = grid.Length;
        int cols = grid[0].Length;

        if (row >= rows || col >= cols){
            return int.MaxValue; // Boundary
        }
    
        if (memo[row][col] != -1){
            return memo[row][col]; // Memoized result
        } 

        if (row == rows - 1 && col == cols - 1){
            return grid[row][col]; // Base case
        } 

        // Recurrance relation        
        memo[row][col] = grid[row][col] + Math.Min(MinSum(grid, memo, row, col + 1), MinSum(grid, memo, row + 1, col));

        return memo[row][col];
    }

    
    public static int MinimumTotal(IList<IList<int>> triangle)
    {
        int n = triangle.Count;
        var memo = new int?[n, n];
        return TriangleSum(triangle, 0, 0, memo);
    }

    private static int TriangleSum(IList<IList<int>> triangle, int row, int col, int?[,] memo) {
        if (row == triangle.Count - 1) {
            return triangle[row][col]; // base case: bottom row
        }

        if (memo[row, col] != null) {  return memo[row, col].Value; }

        int down = TriangleSum(triangle, row + 1, col, memo);
        int diagonal = TriangleSum(triangle, row + 1, col + 1, memo);

        memo[row, col] = triangle[row][col] + Math.Min(down, diagonal);
        return memo[row, col].Value;
    }


    public static int LengthOfLIS(int[] nums)
    {
        List<int> sub = new List<int>();
        foreach (int n in nums)
        {
            int l = 0, r = sub.Count - 1, mid = 0;
            while (l <= r)
            {
                mid = (l + r) / 2;
                if (n <= sub[mid])
                    r = mid - 1;
                else
                    l = mid + 1;
            }
            if (l == sub.Count)
                sub.Add(n);
            else
                sub[l] = n;
            //Console.WriteLine(n+"______" +l+"___"+r+"___"+string.Join(",", sub));
        }
        return sub.Count;
    }
    

    // Recurrence
    // makeChange(amount) = 1 + [ min( makeChange(amount - coin) ) for each coin where coin ≤ amount ]

    public static int CoinChange(int[] coins, int amount)
    {
        int[] memo = new int[amount + 1];
        for (int i = 0; i <= amount; i++)
        {
            memo[i] = -2; // -1 signals no combination
        }
        return MakeChange(coins, amount, memo);
    }

    public static int MakeChange(int[] coins, int amount, int[] memo){   
        if (amount == 0) return 0;     // Base case: no coins needed

        if (memo[amount] != -2) return memo[amount];// Gotcha prune recursive paths

        int min = int.MaxValue;
        foreach (int coin in coins)
        {
            if (amount - coin < 0) continue; // skip invalid subproblem
            int res = MakeChange(coins, amount - coin, memo);
            if (res >= 0 && res < min)
            {
                min = res + 1; // Found a smaller count
            }
        }

        memo[amount] = (min == int.MaxValue) ? -1 : min; // Save in memo
        return memo[amount];
    }


    // Iterative
    // public static int CoinChange(int[] coins, int amount)
    // {
    //     int[] dp = new int[amount + 1];
    //     dp[0] = 0;

    //     for (int i = 1; i <= amount; i++)
    //     {
    //         dp[i] = amount + 1;
    //     }

    //     foreach (int coin in coins)
    //     {
    //         for (int i = coin; i <= amount; i++)
    //         {
    //             dp[i] = Math.Min(dp[i], dp[i - coin] + 1);
    //         }
    //     }

    //     if (dp[amount] == amount + 1)
    //     {
    //         return -1;
    //     }
    //     return dp[amount];
    // }


    public static bool WordBreak(string s, IList<string> wordDict)
    {

        // working backwards from the end to the beginning: "leetcode"
        //                                                      "leet"                             "code"
        // recurrence relation: memo[i] = was there a match from [0->j] and do I have a match from [j->i]  
        // base case: memoization at 0 should be true
        //
        // Nleetcode
        // tffftffft 
        // 
        return BuildWordBreak(s, new HashSet<string>(wordDict), new bool?[s.Length + 1], s.Length);

    }

    private static bool BuildWordBreak(string s, HashSet<string> dict, bool?[] wordbreak, int i){

        if (i == 0){
            return true; // base case has to be set true
        }
        if (wordbreak[i].HasValue) return wordbreak[i].Value; // gotcha to prune recursive paths
        // Was there ever an earlier "hit" and is there a current "hit"?
        for (int j = 0; j < i; j++) {
                // was there a match up to exaclty j  && is there a match from j to i           
            if (BuildWordBreak(s, dict, wordbreak,j) && dict.Contains(s.Substring(j, i - j))) {
                wordbreak[i] = true;
                return true;
            }
        }
        // no words earlier in string exist
        wordbreak[i] = false;
        return false;
    }



    public static int Rob(int[] nums)
    {
        // Recurrence: 
        // sum[i] = Max( sum[i-2], sum[i-3] );  X Wrong!!, forgot to add value from i

        // The "current answer" 
        // Either include 'i" or skip it and the sum up to the previous 'i' 
        // (Can't include i as it's adjacent)
        //
        // sum[i] = Max( sum[i-1], sum[i-2] + nums[i])

        // Base Case
        // sum[0] = nums[0]; 
        // sum[1] = Max( num[1], num[0] )

        var sums = new int[nums.Length];
        Array.Fill(sums, -1);

        return RobHouses(nums, sums, nums.Length - 1);
    }

    private static int RobHouses(int[] nums, int[] sums, int i){
        if (i == 0){ return nums[0]; }
        if (i == 1){ return Math.Max(nums[1], nums[0]); }
        
        if (sums[i] != -1) return sums[i]; // skip filled values to kill recursion

        sums[i] = Math.Max( RobHouses(nums,sums, i-1), RobHouses(nums,sums,i-2) + nums[i] );
        return sums[i];
    }

    //Iterative
    //   public int Rob(int[] nums) {
    //     if (nums == null || nums.Length == 0) return 0;
    //     if (nums.Length == 1) return nums[0];

    //     int prev2 = nums[0];
    //     int prev1 = Math.Max(nums[0], nums[1]);

    //     for (int i = 2; i < nums.Length; i++) {
    //         int curr = Math.Max(prev1, prev2 + nums[i]);
    //         prev2 = prev1;
    //         prev1 = curr;
    //     }

    //     return prev1;
    // }

    // Recursive
    public static int ClimbStairs(int n)
    {
        //                              one more    two more
        // Ways to reach step n = SUM { step n-1 + step n-2}
        // ways[n] = ways[n-1] + ways[n-2]
        // ways[1] = 1 // ways to reach step 1 (only way to reach it is with one step)
        // ways[2] = 2 // ways to reach step 2 (can reach it 2 steps, or 1 step == 2)
        return BuildWays(new int[n + 1], n);
    }

    private static int BuildWays(int[] ways,int i){
        if (i == 1){  return 1; }
        if (i == 2){  return 2; }
        
        if (ways[i] != 0) return ways[i]; // don't recompute the same one

        ways[i] = BuildWays(ways, i-1) + BuildWays(ways,i-2);
        
        return ways[i];
    }
    // Iterative
    // public int ClimbStairs(int n) {
    //     if (n <= 2) return n;

    //     var ways = new int[n + 1];
    //     ways[1] = 1;
    //     ways[2] = 2;

    //     for (int i = 3; i <= n; i++) {
    //         ways[i] = ways[i - 1] + ways[i - 2];
    //     }

    //     return ways[n];
    // }
    

       // y = mx+b
    // gotcha: Repeated points won't create a line
    public static int MaxPoints(int[][] points) {
        int maxFitPoints = 0;
        for (int i=0; i< points.Length; i++){
            int x1 = points[i][0];
            int y1 = points[i][1];
            int localMax = 0;
            var hashCount = new Dictionary<(int, int), int>();  
            int dupes = 0;            

            for (int j=i+1; j< points.Length; j++){
                int x2 = points[j][0];
                int y2 = points[j][1];
                int dy = y2-y1;
                int dx = x2-x1;
        
                if (dx == 0 && dy == 0) {
                    dupes++;
                    continue;
                }

                // Gotcha! Don't do this!
                // for values of m + b that are the same, they fit the line, add to the count;
                // int m = (y2-y1)/(x2-x1); // This is a gotcha, for decimal precision and edge cases
                // int b = y1-(m*x1); 
                int gcd = GCD(dy,dx); // keep as simplified fractions

                dy = dy/gcd;
                dx = dx/gcd;

                if (dx == 0) {
                    dy = 1; dx = 0; // vertical line
                } else if (dy == 0) {
                    dx = 1; dy = 0; // horizontal line
                } else if (dx < 0) {
                    //Original slope	After normalization
                    //(1,1)	            (1,1)
                    //(-1,-1)	        (1,1)
                    //(1,-1)	        (1,-1)
                    //(-1,1)	        (1,-1)
                    
                    // Example
                    // (0,0) -> (-3,2) has slope 2/-3 = -2/3
                    // (0,0) -> (3,-2) has slope -2/3                    
                    dx = -dx; dy = -dy;     // keep dx positive doesn't affect the slope
                }

                var slope = (dy,dx);

                if (!hashCount.ContainsKey(slope)) { hashCount[slope] = 0; }
                hashCount[slope]++;
                
                if (hashCount[slope] > localMax) {
                    localMax = hashCount[slope];
                }
            }
            maxFitPoints = Math.Max(maxFitPoints, localMax + dupes + 1);
        }
        return maxFitPoints;
    }

     // Gleaned a GCD method
     private static int GCD(int a, int b) {
        a = Math.Abs(a);
        b = Math.Abs(b);
        while (b != 0) {
            int tmp = b;
            b = a % b;
            a = tmp;
        }
        return a;
    }

    
      public static int MySqrt(int x)
    {
        // Approach 1
        // simple brute force idea -- linear search
        // keep incremeneting and double to find perfect square less than or 
        // equal to x

        // Approach 2
        // use binary search rather than linear search  -- optimized
        //
        //  1         x
        // left  mid  right
        int left = 1, right = x, ans = 0;
        while (left <= right)
        {
            int mid = left + (right - left) / 2;
            long square = (long)mid * mid; // gotcha, long to avoid overflows

            if (square == x)
            {
                return mid;
            }
            else if (square < x)
            {
                ans = mid;
                left = mid + 1;
            }
            else
            {
                right = mid - 1;
            }
        }
        return ans;  // floor of sqrt(x)
    }

    public static int TrailingZeroes(int n)
    {
        // You are not supposed to multiply the factoral then divide powers of 10.
        // Numbers will get too large
        //
        // Count the factors of 5

        // e.g. 25
        // 5,10,15,20,25 --> 6 factors
        // 25*24*23*22*21*20*19*18*17*16*15*14...1
        //    (2)   (2)   (2)   (2)  (2)  (2)... etc (at least 6 of them)
        //     *     *     *    *    *     * 
        //    5*2   10(5*2)  15(5*3) 20(5*4) 25(5*5)... etc.. A(6 5's)
        //    10  *  10  *   10 *   10 *  10 *  10  --Pull out the factor of 5 multiply by any 2 available
        //   1000000 (6 zeros)

        // how many 5's?
        // e.g. 25
        // 25/5 = 5
        // 5/5 =  1
        // 6 5's in total
        int count = 0;
        while (n > 0)
        {
            n = n / 5;
            count += n;
        }
        return count;
    }

    public static int[] PlusOne(int[] digits)
    {
        for (int i = digits.Length - 1; i >= 0; i--)
        {
            if (digits[i] < 9)
            {
                digits[i]++;
                // [9,9,9,9,9,8] ->  [9,9,9,9,9,9] -> exit
                return digits;
            }
            // [1,9,9,9,9,9] -> [1,0,0,0,0,0] -> Next iteration it would reach if{} statement -> [2,0,0,0,0,0] -> exit
            digits[i] = 0;  // carry over
        }

        // can only reach here if it's all 9's
        // [9,9,9,9,9] -> [1,0,0,0,0,0] -> exit
        int[] result = new int[digits.Length + 1];
        result[0] = 1;
        return result;
    }


    public static bool IsPalindrome(int x)
    {
        if (x < 0)
        {
            return false;
        }

        // Find the divisor to extract the most significant digit
        int div = 1;
        while (x / div >= 10)
        {
            div *= 10;
        }

        while (x > 0)
        {
            int msb = x / div;
            int lsb = x % 10;
            if (msb != lsb) return false;

            x = (x % div) / 10;
            div = div / 100; // two numbers are gone not just one.
        }
        return true;
    }

    // standard answer
    //public bool IsPalindrome(int x) {
    //    if (x < 0 || (x % 10 == 0 && x != 0)) return false;
    //
    //    int reversedHalf = 0;
    //    while (x > reversedHalf) {
    //        reversedHalf = reversedHalf * 10 + x % 10;
    //        x /= 10;
    //    }
    //    // Even length: x == reversedHalf
    //    // Odd length: x == reversedHalf / 10
    //    return x == reversedHalf || x == reversedHalf / 10;
    //}


    public static int SingleNumberTriplets(int[] nums) {
    // 1010
    // 1010  
    // 1010
    // 1000 <--- save

    // Looping through all 32 bits with a bitmask through all numbers. 
    // Store a bitcounter of three- iterate bit-by-bit, if you come across three of them, cancel out. 
    // Then reset the triplet counter to 0. Anything left in the counter by the end (should either be three or 1 or 0), 
    // you position that bit in place

    // Better to have a modulo 3 that sets the bit
        int result =0;
        for (int j=0; j< 32; j++){
            int bit =0;
            int bitmask = 1 << j;
            long bitresult = 0;

            for (int i=0; i< nums.Length; i++){
                int num = nums[i];
                
                bitresult += (nums[i] & bitmask);
            }

            // If this bit contributes to the unique number
            if ((bitresult % 3) != 0) {
                result = result | bitmask;
            }
        }
        return result;

    }


    //  1011  - same
    //  1000 - diff
    //  1011 - same

    //  1011 - same
    // ^1000  -diff
    // ------
    //  0011
    //
    //   0011
    //  ^1011 - same
    // -----
    //  1000 - diff -- yay
    //
    public static int SingleNumber(int[] nums)
    {

        int result = 0;

        for (int i = 0; i < nums.Length; i++)
        {
            result = result ^ nums[i];
        }
        return result;

    }

    public static int HammingWeight(int n)
    {
        int weight = 0;
        while (n > 0)
        {
            if ((n & 1) == 1)
            {
                weight++;
            }
            n = n >> 1; // shift right
        }
        return weight;
    }

    public static int ReverseBitsSimplier(int n)
    {
        var res = 0;
        for (int i = 0; i < 32; i++)
        {
            var bit = (n >> i) & 1;      // extract the i-th bit
            res = res | (bit << (31 - i)); // put it in reversed position
        }
        return res;
    }

    // input 0101 -> output 1010
    //
    //  0101 (input)
    // &   1 
    //  0001 (mask)
    //
    //  0000 (result)
    // |0001 (mask)
    //  0001 (result)
    //
    //  0101 (input)
    // >>  1 (shift right)
    //   010 (new input)
    //
    //  Next iteration
    //  0001 (result)
    // <<  1 (shift left)
    //  0010 (result)
    //
    // ...etc
    public static int ReverseBits(int n)
    {
        uint x = (uint)n;
        uint result = 0;
        for (int i = 0; i < 32; i++)
        {
            result = result << 1; // shift left to make the digit more significant
            uint mask = x & 1; // e.g. 0000001 to get the last digit
            result = result | mask; // use an OR to preserve the last digit
            x = x >> 1; // shift over and the rightmost bit falls off            
        }
        return (int)result;
    }

    public static string AddBinary(string a, string b)
    {

        // right to left simply carry over to next significat digit

        StringBuilder answer = new StringBuilder();

        int indexA = a.Length - 1;
        int indexB = b.Length - 1;
        int carryOver = 0;

        while (indexA >= 0 || indexB >= 0 || carryOver > 0)
        {

            int aBit = 0, bBit = 0, sum = 0;

            if (indexA >= 0)
            {
                aBit = a[indexA] - '0'; // can't cast to (int)
                indexA--;
            }

            if (indexB >= 0)
            {
                bBit = b[indexB] - '0'; // can't cast to (int)
                indexB--;
            }

            sum = (carryOver + aBit + bBit) % 2; // set the BIT not the int
            carryOver = (carryOver + aBit + bBit) / 2;  // carry over the Bit not the int

            answer.Append(sum); // this is going to be have the answer in reverse order
        }

        // Java has .reverse() in StringBuffer, C# you have to do this nonsense
        var result = answer.ToString().ToCharArray();
        Array.Reverse(result);
        var reversed = new string(result);

        return reversed;
    }



    // Min Heap - All projects ordered by least expensive to most expensive
    // Max Heap - Most Profitable Projects - add from min heap as we go only based on capital 
    public static int FindMaximizedCapital(int k, int w, int[] profits, int[] capital)
    {
        var capitalMinHeap = new PriorityQueue<(int cap, int profit), int>();
        var profitMaxHeap = new PriorityQueue<int, int>();

        for (int i = 0; i < profits.Length; i++)
        {
            capitalMinHeap.Enqueue((capital[i], profits[i]), capital[i]);
        }

        // each iteration we keep adding all the projects we can afford 
        // grabbing the most profitable one
        for (int i = 0; i < k; i++)
        {
            while (capitalMinHeap.Count > 0 && capitalMinHeap.Peek().cap <= w)
            {
                var project = capitalMinHeap.Dequeue();
                profitMaxHeap.Enqueue(project.profit, project.profit * -1);
            }

            if (profitMaxHeap.Count == 0) break;

            w += profitMaxHeap.Dequeue();
        }

        return w;
    }


    public static int FindKthLargestOptimal(int[] nums, int k)
    {
        var pq = new PriorityQueue<int, int>();

        foreach (int num in nums)
        {
            pq.Enqueue(num, num); // small high priority
            if (pq.Count > k)
            {
                pq.Dequeue(); // this will remove the smallest ones first
            }
        }

        return pq.Peek(); // returns the smallest == k'th largest
    }

    public static double FindMedianSortedArrays(int[] nums1, int[] nums2)
    {
        if (nums1.Length > nums2.Length)
            return FindMedianSortedArrays(nums2, nums1);

        int m = nums1.Length, n = nums2.Length;
        int left = 0, right = m;

        while (left <= right)
        {
            int mid1 = (left + right) / 2;
            int mid2 = (m + n + 1) / 2 - mid1;

            int l1 = (mid1 == 0) ? int.MinValue : nums1[mid1 - 1];
            int r1 = (mid1 == m) ? int.MaxValue : nums1[mid1];
            int l2 = (mid2 == 0) ? int.MinValue : nums2[mid2 - 1];
            int r2 = (mid2 == n) ? int.MaxValue : nums2[mid2];

            if (l1 <= r2 && l2 <= r1)
            {
                int maxLeft = Math.Max(l1, l2);
                int minRight = Math.Min(r1, r2);
                return ((m + n) % 2 == 0)
                    ? (maxLeft + minRight) / 2.0
                    : maxLeft;
            }
            else if (l1 > r2)
            {
                right = mid1 - 1;
            }
            else
            {
                left = mid1 + 1;
            }
        }

        throw new ArgumentException("Input arrays not sorted");
    }

    public static int FindMin(int[] nums)
    {
        int left = 0;
        int right = nums.Length - 1;

        while (left < right)
        {
            int mid = left + (right - left) / 2;

            // min must be in the right half
            if (nums[mid] > nums[right])
            {
                left = mid + 1;
            }
            else
            {
                // min is in left half (including mid)
                right = mid;
            }
        }

        return nums[left];
    }

    public static int[] SearchRange(int[] nums, int target)
    {
        int first = FindEnd(nums, target, true);
        int last = FindEnd(nums, target, false);
        return new int[] { first, last };
    }
    private static int FindEnd(int[] nums, int target, bool searchleft)
    {
        int left = 0;
        int right = nums.Length - 1;
        int bound = -1;

        while (left <= right)
        {
            int mid = left + (right - left) / 2;

            if (nums[mid] == target)
            {
                bound = mid;
                if (searchleft)
                {
                    // keep looking left
                    right = mid - 1;
                }
                else
                {
                    // keep looking right
                    left = mid + 1;
                }
            }
            else if (nums[mid] < target)
            {
                left = mid + 1;
            }
            else
            {
                right = mid - 1;
            }
        }
        return bound;
    }

    public static int FindPeakElement(int[] nums)
    {
        int left = 0;
        int right = nums.Length - 1;

        while (left < right)
        {
            int mid = left + (right - left) / 2;

            int prev = mid - 1 >= 0 ? nums[mid - 1] : int.MinValue;
            int next = mid + 1 < nums.Length ? nums[mid + 1] : int.MinValue;

            // Peak case: greater than both neighbors
            if (nums[mid] > prev && nums[mid] > next)
            {
                return mid;
            }

            // upward slope - peak to the right
            if (nums[mid] < nums[mid + 1])
            {
                left = mid + 1;
            }
            else
            {
                // downward slope - peak is to the left
                right = mid;
            }
        }

        return left; // only edge cases should reach here 
    }

    public static bool SearchMatrix(int[][] matrix, int target)
    {
        int m = matrix.Length;
        int n = matrix[0].Length;

        int left = 0;
        int right = m * n - 1;

        while (left <= right)
        {
            int mid = left + (right - left) / 2;
            int row = mid / n;
            int column = mid % n;
            int val = matrix[row][column];

            if (target == val) { return true; }

            if (target < val) { right = mid - 1; }
            else { left = mid + 1; }
        }
        return false;
    }


    public static int SearchInsert(int[] nums, int target)
    {
        int left = 0;
        int right = nums.Length - 1;

        while (right >= left)
        {
            int half = left + (right - left) / 2;

            if (nums[half] == target)
            {
                return half;
            }

            if (nums[half] < target)
            {
                left = half + 1;
            }
            else
            {
                right = half - 1;
            }
        }
        return left;
    }


    public static int MaxSubarraySumCircular(int[] nums)
    {
        int total = 0;
        int maxSum = nums[0];
        int curMax = 0;
        int minSum = nums[0];
        int curMin = 0;

        foreach (int num in nums)
        {
            curMax = Math.Max(num, curMax + num);
            maxSum = Math.Max(maxSum, curMax);

            curMin = Math.Min(num, curMin + num);
            minSum = Math.Min(minSum, curMin);

            total += num;
        }

        if (maxSum < 0) return maxSum;
        return Math.Max(maxSum, total - minSum);
    }


    public static ListNode MergeKLists(ListNode[] lists)
    {
        var pq = new PriorityQueue<ListNode, int>();
        foreach (var node in lists)
        {
            if (node != null)
                pq.Enqueue(node, node.val);
        }

        ListNode dummy = new ListNode(0);
        ListNode current = dummy;

        while (pq.Count > 0)
        {
            var node = pq.Dequeue();
            current.next = node;
            current = current.next;

            if (node.next != null)
                pq.Enqueue(node.next, node.next.val);
        }

        return dummy.next;
    }
    public static QNode Construct(int[][] grid)
    {
        int n = grid.Length;
        return BuildRecursively(grid, 0, 0, n);
    }
    private static QNode BuildRecursively(int[][] grid, int row, int col, int size)
    {
        // Even if it's a huge matrix of all the values
        // you want to have a leaf node output here
        if (IsUniform(grid, row, col, size))
        {
            return new QNode(grid[row][col] == 1, true);
        }
        int half = size / 2;
        QNode topLeft = BuildRecursively(grid, row, col, half);
        QNode topRight = BuildRecursively(grid, row, col + half, half);
        QNode bottomLeft = BuildRecursively(grid, row + half, col, half);
        QNode bottomRight = BuildRecursively(grid, row + half, col + half, half);
        return new QNode(true, false, topLeft, topRight, bottomLeft, bottomRight);
    }

    private static bool IsUniform(int[][] grid, int row, int col, int size)
    {
        int val = grid[row][col];
        for (int i = row; i < row + size; i++)
        {
            for (int j = col; j < col + size; j++)
            {
                if (grid[i][j] != val) return false;
            }
        }
        return true;
    }

    public static ListNode SortList(ListNode head)
    {
        List<int> storage = new List<int>();
        while (head != null)
        {
            storage.Add(head.val);
            head = head.next;
        }
        storage.Sort();
        ListNode dummy = new ListNode();
        ListNode current = dummy;
        foreach (int item in storage)
        {
            current.next = new ListNode(item);
            current = current.next;
        }
        return dummy.next;
    }

    public static bool Exist(char[][] board, string word)
    {
        for (int i = 0; i < board.Length; i++)
        {
            for (int j = 0; j < board[0].Length; j++)
            {
                if (Combine(board, word, 0, i, j))
                    return true;
            }
        }
        return false;
    }

    private static bool Combine(char[][] board, string word, int index, int row, int col)
    {
        if (index == word.Length) return true;
        if (row < 0 || row >= board.Length || col < 0 || col >= board[0].Length || board[row][col] != word[index])
            return false;
        char temp = board[row][col];
        board[row][col] = '#';
        bool found = Combine(board, word, index + 1, row + 1, col) ||
                     Combine(board, word, index + 1, row - 1, col) ||
                     Combine(board, word, index + 1, row, col + 1) ||
                     Combine(board, word, index + 1, row, col - 1);
        board[row][col] = temp;
        return found;
    }

    static List<string> validParens = new List<string>();
    public static IList<string> GenerateParenthesis(int n)
    {
        Combinations("", 0, 0, n);
        return validParens;
    }

    public static void Combinations(string parens, int numLeft, int numRight, int n)
    {
        if (parens.Length == (n * 2))
        {
            validParens.Add(parens);
            // No need for post-validation because 
            // we constructed the parens based 
            // on rules below.
            return;
        }

        // backtracking because we are re-using the same
        // parens after calling Combinations()

        // never add more '(' than the max
        if (numLeft < n)
        {
            Combinations(parens + '(', numLeft + 1, numRight, n);
        }

        // never add a ')' more than we have ')'
        if (numRight < numLeft)
        {
            Combinations(parens + ')', numLeft, numRight + 1, n);
        }
    }

    static int count = 0;

    public static int TotalNQueens(int n)
    {
        Combine(new HashSet<int>(), new HashSet<int>(), new HashSet<int>(), 0, n);
        return count;
    }
    public static void Combine(HashSet<int> colPlacement, HashSet<int> posSlopeIntercept, HashSet<int> negSlopeIntercept, int x, int n)
    {
        if (colPlacement.Count == n)
        {
            count++;
        }
        // cols
        for (int y = 0; y < n; y++)
        {
            if (colPlacement.Contains(y) || posSlopeIntercept.Contains(x - y) || negSlopeIntercept.Contains(x + y))
            {
                continue;
            }
            colPlacement.Add(y);
            posSlopeIntercept.Add(x - y);
            negSlopeIntercept.Add(x + y);

            Combine(colPlacement, posSlopeIntercept, negSlopeIntercept, x + 1, n);

            colPlacement.Remove(y);
            posSlopeIntercept.Remove(x - y);
            negSlopeIntercept.Remove(x + y);
        }
    }


    public class Solution
    {

        public IList<IList<int>> CombinationSum(int[] candidates, int target)
        {
            Combinations(candidates, target, 0, new List<int>());
            return retval;
        }

        public void Combinations(int[] candidates, int target, int start, List<int> combination)
        {
            int sum = 0;
            foreach (var val in combination) sum += val;
            if (sum > target) return;
            if (sum == target)
            {
                retval.Add(new List<int>(combination));
                return;
            }

            for (int i = start; i < candidates.Length; i++)
            {
                combination.Add(candidates[i]);
                Combinations(candidates, target, i, combination);
                combination.RemoveAt(combination.Count - 1);
            }
        }

        public IList<IList<int>> Permute(int[] nums)
        {
            Combine(new List<int>(), nums, new bool[nums.Length]);
            return retval;
        }

        public void Combine(List<int> combination, int[] nums, bool[] used)
        {

            if (combination.Count == nums.Length)
            {
                retval.Add(new List<int>(combination));
            }

            for (int i = 0; i < nums.Length; i++)
            {
                if (used[i]) continue;

                used[i] = true;
                combination.Add(nums[i]);
                Combine(combination, nums, used);
                combination.RemoveAt(combination.Count - 1);
                used[i] = false;
            }
        }

        // 1 - 1 - 1 x
        // 1 - 1 - 2 x
        // 1 - 1 - 3 x
        // 1 - 2 - 1 x
        // 1 - 2 - 2 x
        // 1 - 2 - 3 
        // 1 - 3 - 1 x
        // 1 - 3 - 2 x
        // 1 - 3 - 3 x
        // 2 - 1 - 1 x
        // 2 - 1 - 2 x
        // 2 - 1 - 3 x
        // 2 - 2 - 1 x
        // 2 - 2 - 2 x
        // 2 - 2 - 3 x
        private List<IList<int>> retval = new List<IList<int>>();

        public IList<IList<int>> Combine(int n, int k)
        {
            Combinations(new List<int>(), 1, n, k);
            return retval;
        }

        private void Combinations(List<int> combination, int start, int n, int k)
        {
            if (combination.Count == k)
            {
                // reached the lengh of choices
                retval.Add(new List<int>(combination));
                return;
            }
            for (int i = start; i <= n; i++)
            {
                combination.Add(i); // [1] --> [2] -> [3]  || [1,2] -> [1,3]
                Combinations(combination, i + 1, n, k); // [1] --> [2] -> [3] || [1,2] -> [1,3]
                combination.RemoveAt(combination.Count - 1);  // [] -> [] -> []  || [1] -> [1]
            }
        }
    }

    static Dictionary<char, char[]> dictionary = new Dictionary<char, char[]>();

    public static IList<string> LetterCombinations(string digits)
    {
        if (string.IsNullOrEmpty(digits)) return returnList;

        dictionary.Add('2', new char[] { 'a', 'b', 'c' });
        dictionary.Add('3', new char[] { 'd', 'e', 'f' });
        dictionary.Add('4', new char[] { 'g', 'h', 'i' });
        dictionary.Add('5', new char[] { 'j', 'k', 'l' });
        dictionary.Add('6', new char[] { 'm', 'n', 'o' });
        dictionary.Add('7', new char[] { 'p', 'q', 'r', 's' });
        dictionary.Add('8', new char[] { 't', 'u', 'v' });
        dictionary.Add('9', new char[] { 'w', 'x', 'y', 'z' });
        chainResponse("", digits, 0);
        return returnList;
    }

    public static void chainResponse(string currword, string digits, int i)
    {
        if (i == digits.Length)
        {
            returnList.Add(currword);
            return;
        }
        char number = digits[i];
        char[] letters = dictionary[number];

        for (int j = 0; j < letters.Length; j++)
        {
            chainResponse(currword + letters[j], digits, i + 1);
        }
    }

    static List<string> returnList = new List<string>();
    static TrieNode root = new TrieNode();

    public static IList<string> FindWords(char[][] board, string[] words)
    {
        // 1. Build up dictionary words as a trie
        foreach (var word in words)
        {
            TrieNode curr = root;
            foreach (var c in word)
            {
                if (!curr.Children.ContainsKey(c))
                    curr.Children[c] = new TrieNode(c);
                curr = curr.Children[c];
            }
            curr.isEndofWord = true;
        }


        // 2. Across the board, pass trie and board
        for (int i = 0; i < board.Length; i++)
        {
            for (int j = 0; j < board[0].Length; j++)
            {
                BuildWord(board, i, j, root, "");
            }
        }

        return returnList;

    }

    public static void BuildWord(char[][] board, int i, int j, TrieNode node, string path)
    {
        if (i < 0 || j < 0 || i >= board.Length || j >= board[0].Length || board[i][j] == '#')
            return;

        char c = board[i][j];
        if (!node.Children.ContainsKey(c))
            return;

        node = node.Children[c];
        string newPath = path + c;

        if (node.isEndofWord)
        {
            returnList.Add(newPath);
            node.isEndofWord = false;
        }

        board[i][j] = '#'; // mark visited
        BuildWord(board, i + 1, j, node, newPath); // down
        BuildWord(board, i - 1, j, node, newPath); // up
        BuildWord(board, i, j + 1, node, newPath); // right
        BuildWord(board, i, j - 1, node, newPath); // left
        board[i][j] = c; // restore

    }

    public class WordDictionary
    {
        private TrieNode root = new TrieNode();

        public void AddWord(string word)
        {
            TrieNode current = root;
            foreach (var c in word)
            {
                if (!current.Children.ContainsKey(c))
                {
                    TrieNode node = new TrieNode();
                    node.c = c;
                    current.Children[c] = node;
                }
                current = current.Children[c];
            }
            current.isEndofWord = true;
        }
        public bool Search(string word)
        {
            return SearchInNode(word, 0, root);
        }
        private bool SearchInNode(string word, int index, TrieNode node)
        {
            if (index == word.Length) return node.isEndofWord;
            char c = word[index];
            if (c == '.')
            {
                // This should go down all possible paths rather than a single path
                foreach (var child in node.Children.Values)
                {
                    if (SearchInNode(word, index + 1, child))
                    {
                        return true;
                    }
                }
                return false;
            }
            else
            {
                if (!node.Children.ContainsKey(c)) return false;
                return SearchInNode(word, index + 1, node.Children[c]);
            }
        }
    }

    public class TrieNode
    {
        public TrieNode()
        {
        }
        public TrieNode(char c)
        {
            this.c = c;
        }
        public bool isEndofWord { get; set; }
        public char c { get; set; }
        public Dictionary<char, TrieNode> Children = new Dictionary<char, TrieNode>();
    }


    public class Trie
    {
        TrieNode root;
        public Trie()
        {
            root = new TrieNode();
        }

        public void Insert(string word)
        {
            TrieNode current = root;
            foreach (char c in word)
            {
                if (!current.Children.ContainsKey(c))
                {
                    current.Children[c] = new TrieNode(c);
                }
                current = current.Children[c];
            }
            current.isEndofWord = true;
        }

        public bool Search(string word)
        {
            TrieNode current = root;
            foreach (char c in word)
            {
                if (!current.Children.ContainsKey(c))
                {
                    return false;
                }
                current = current.Children[c];
            }
            return current.isEndofWord;
        }

        public bool StartsWith(string prefix)
        {
            TrieNode current = root;
            foreach (char c in prefix)
            {
                if (!current.Children.ContainsKey(c))
                {
                    return false;
                }
                current = current.Children[c];
            }
            return true;
        }
    }

    public static int LadderLength(string beginWord, string endWord, IList<string> wordList)
    {
        HashSet<string> bank = new HashSet<string>(wordList);
        HashSet<string> visited = new HashSet<string>();
        char[] chars = new char[] { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z' };
        Queue<(string, int)> q = new Queue<(string, int)>();
        q.Enqueue((beginWord, 1));
        while (q.Count > 0)
        {
            (string word, int count) = q.Dequeue();
            if (word == endWord)
            {
                return count;
            }
            for (int i = 0; i < word.Length; i++)
            {
                foreach (char c in chars)
                {
                    StringBuilder sb = new StringBuilder(word);
                    sb[i] = c;

                    string next = sb.ToString();

                    if (bank.Contains(next) && !visited.Contains(next))
                    {
                        visited.Add(next);
                        q.Enqueue((next, count + 1));
                    }
                }
            }
        }
        return 0;
    }

    public static int MinMutation(string startGene, string endGene, string[] bank)
    {
        HashSet<string> visited = new HashSet<string>();
        HashSet<string> geneBank = new HashSet<string>(bank);
        Queue<(string, int)> q = new Queue<(string, int)>();
        q.Enqueue((startGene, 0));
        while (q.Count > 0)
        {
            var (gene, count) = q.Dequeue();
            if (gene == endGene)
            {
                return count;
            }
            for (int i = 0; i < gene.Length; i++)
            {
                foreach (char c in new char[] { 'A', 'C', 'G', 'T' })
                {
                    var newGene = Substitute(gene, i, c);
                    if (geneBank.Contains(newGene) && !visited.Contains(newGene))
                    {
                        visited.Add(newGene);
                        q.Enqueue((newGene, count + 1));
                    }
                }
            }
        }
        return -1;
    }
    private static string Substitute(string gene, int index, char c)
    {
        StringBuilder sb = new StringBuilder(gene);
        sb[index] = c;
        return sb.ToString();
    }


    public static int SnakesAndLadders(int[][] board)
    {
        int n = board.Length;
        var visited = new bool[n * n + 1];
        var queue = new Queue<(int pos, int moves)>();
        queue.Enqueue((1, 0)); // start at square 1, 0 moves
        visited[1] = true;
        while (queue.Count > 0)
        {
            var (curr, moves) = queue.Dequeue();
            for (int i = 1; i <= 6; i++)
            {
                int next = curr + i;
                if (next > n * n) continue;

                var (row, col) = GetCoordinates(next, n);
                if (board[row][col] != -1)
                    next = board[row][col]; // snake or ladder

                if (next == n * n)
                    return moves + 1;

                if (!visited[next])
                {
                    visited[next] = true;
                    queue.Enqueue((next, moves + 1));
                }
            }
        }
        return -1;
    }
    private static (int row, int col) GetCoordinates(int square, int n)
    {
        int r = (square - 1) / n;
        int c = (square - 1) % n;
        int row = n - 1 - r;
        int col = (r % 2 == 0) ? c : n - 1 - c;
        return (row, col);
    }


    public static List<int> result = new List<int>();

    public static int[] FindOrder(int numCourses, int[][] prerequisites)
    {
        var graph = new Dictionary<int, List<int>>();

        foreach (var edge in prerequisites)
        {
            int from = edge[1], to = edge[0];
            if (!graph.ContainsKey(from)) graph[from] = new List<int>();
            graph[from].Add(to);
        }
        var state = new Dictionary<int, int>(); // 0 = unvisited, 1 = visiting, 2 = visited       
        for (int i = 0; i < numCourses; i++)
        {
            if (!state.ContainsKey(i))
            {
                if (HasCycle(i, graph, state))
                {
                    return []; // Cycle found
                }
            }
        }
        result.Reverse();
        return result.ToArray();
    }

    private static bool HasCycle(int node, Dictionary<int, List<int>> graph, Dictionary<int, int> state)
    {
        state[node] = 1; // mark as visiting
        if (graph.ContainsKey(node))
        {
            foreach (var neighbor in graph[node])
            {
                if (!state.ContainsKey(neighbor))
                {
                    if (HasCycle(neighbor, graph, state)) return true;
                }
                else if (state[neighbor] == 1)
                {
                    // Cycle detected
                    return true;
                }
            }
        }
        state[node] = 2; // mark as visited
        result.Add(node);
        return false;
    }


    public static bool CanFinish(int numCourses, int[][] prerequisites)
    {
        var graph = new Dictionary<int, List<int>>();
        foreach (var edge in prerequisites)
        {
            int from = edge[1], to = edge[0];
            if (!graph.ContainsKey(from)) graph[from] = new List<int>();
            graph[from].Add(to);
        }
        var state = new Dictionary<int, int>(); // 0 = unvisited, 1 = visiting, 2 = visited
        foreach (var node in graph.Keys)
        {
            if (!state.ContainsKey(node) && HasCycleDFS(node, graph, state))
            {
                return false; // Cycle found cannot finish
            }
        }
        return true; // No cycle found, can finish
    }
    private static bool HasCycleDFS(int node, Dictionary<int, List<int>> graph, Dictionary<int, int> state)
    {
        state[node] = 1; // mark as visiting
        if (graph.ContainsKey(node))
        {
            foreach (var neighbor in graph[node])
            {
                if (!state.ContainsKey(neighbor))
                {
                    if (HasCycleDFS(neighbor, graph, state)) return true;
                }
                else if (state[neighbor] == 1)
                {
                    // Cycle detected
                    return true;
                }
            }
        }
        state[node] = 2; // mark as visited
        return false;
    }

    public static double[] CalcEquation(IList<IList<string>> equations, double[] values, IList<IList<string>> queries)
    {
        // Example
        // a/b = 2
        // b/c = 3

        // a = 6c
        // b = 3c
        // c = c

        // a --2.0-> b 
        // b --0.5-> a
        // b --3.0-> c 
        // c --1/3-> b

        // add nodes to dictionary, follow links 

        // Build up the dictionary
        Dictionary<string, List<Edge>> graph = new Dictionary<string, List<Edge>>();

        int length = values.Length;

        for (int i = 0; i < equations.Count; i++)
        {
            string a = equations[i][0];
            string b = equations[i][1];
            double value = values[i];

            if (!graph.ContainsKey(a)) graph[a] = new List<Edge>();
            if (!graph.ContainsKey(b)) graph[b] = new List<Edge>();

            graph[a].Add(new Edge { Variable = b, Weight = value });
            graph[b].Add(new Edge { Variable = a, Weight = 1.0 / value });
        }

        List<double> result = new List<double>();

        foreach (var query in queries)
        {
            string start = query[0];
            string end = query[1];

            if (!graph.ContainsKey(start) || !graph.ContainsKey(end))
            {
                result.Add(-1.0);
            }
            else
            {
                HashSet<string> visited = new HashSet<string>();
                double val = Dfs(graph, start, end, 1.0, visited);
                result.Add(val);
            }
        }

        return result.ToArray();
    }

    private static double Dfs(Dictionary<string, List<Edge>> graph, string current, string target, double accProduct, HashSet<string> visited)
    {
        if (current == target) return accProduct;

        visited.Add(current);

        foreach (var neighbor in graph[current])
        {
            if (!visited.Contains(neighbor.Variable))
            {
                double result = Dfs(graph, neighbor.Variable, target, accProduct * neighbor.Weight, visited);
                if (result != -1.0)
                    return result;
            }
        }

        return -1.0;
    }



    private static Dictionary<int, GraphNode> map = new Dictionary<int, GraphNode>();

    public static GraphNode CloneGraph(GraphNode node)
    {
        if (node == null) return null;

        // return visited one
        if (map.ContainsKey(node.val))
            return map[node.val];

        GraphNode copy = new GraphNode(node.val);
        map[node.val] = copy;

        if (node.neighbors != null)
        {
            foreach (GraphNode neighbour in node.neighbors)
            {
                copy.neighbors.Add(CloneGraph(neighbour));
            }
        }
        return copy;

    }

    public static void MarkVisited(char[][] board, int i, int j, int rows, int cols)
    {
        if (i > rows - 1 || j > cols - 1 || j < 0 || i < 0 || board[i][j] != 'O')
        {
            return;
        }

        board[i][j] = 'V';
        MarkVisited(board, i, j + 1, rows, cols);
        MarkVisited(board, i, j - 1, rows, cols);
        MarkVisited(board, i + 1, j, rows, cols);
        MarkVisited(board, i - 1, j, rows, cols);
    }

    public static void Solve(char[][] board)
    {

        // I understand you just flood the region with 0's as long as they are not on the edge
        int rows = board.Length;
        int cols = board[0].Length;

        // 1. mark all border '0's as 'V'
        // recurse through their connected nodes as 'V'

        for (int i = 0; i < rows; i++)
        {
            MarkVisited(board, i, 0, rows, cols);
            MarkVisited(board, i, cols - 1, rows, cols);
        }

        for (int j = 0; j < cols; j++)
        {
            MarkVisited(board, 0, j, rows, cols);
            MarkVisited(board, rows - 1, j, rows, cols);
        }

        // 2. visit all nodes with '0' mark as 'X'        
        // 3. mark 'V's as 'X'

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                // these are not 'V' and should be surrounded by 'X'
                if (board[i][j] == 'O')
                {
                    board[i][j] = 'X';
                }
                else if (board[i][j] == 'V')
                { // revert these back to 'O'
                    board[i][j] = 'O';
                }
            }
        }


    }


    public static int NumIslands(char[][] grid)
    {
        if (grid == null || grid.Length == 0 || grid[0].Length == 0) return 0;

        int rows = grid.Length;
        int columns = grid[0].Length;
        int count = 0;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                if (grid[i][j] == '1')
                {
                    flood(grid, i, j, rows, columns);
                    count++;
                }
            }
        }
        return count;
    }

    public static void flood(char[][] grid, int i, int j, int rows, int columns)
    {
        if (i < 0 || i > rows - 1 || j < 0 || j > columns - 1 || grid[i][j] == '0')
        {
            return;
        }
        grid[i][j] = '0';
        // we are here because something was a "1"
        // anything touching a 1 should be reset to 0 in all directions
        flood(grid, i + 1, j, rows, columns);
        flood(grid, i - 1, j, rows, columns);
        flood(grid, i, j + 1, rows, columns);
        flood(grid, i, j - 1, rows, columns);
    }


    public static bool IsValidBST(TreeNode root)
    {
        Stack<TreeNode> stack = new Stack<TreeNode>();
        TreeNode curr = root;
        TreeNode prev = null;

        while (curr != null || stack.Count > 0)
        {
            while (curr != null)
            {
                stack.Push(curr);
                curr = curr.left;
            }
            curr = stack.Pop();

            if (prev != null && prev.val >= curr.val)
            {
                // sommething off with the binary tree, not increasing in values
                return false;
            }
            prev = curr;
            curr = curr.right;
        }
        return true;
    }

    public static int KthSmallest(TreeNode root, int k)
    {
        Stack<TreeNode> stack = new Stack<TreeNode>();
        int visitCount = 0;
        TreeNode current = root;

        while (current != null || stack.Count > 0)
        {
            // LEFT
            if (current != null)
            {
                while (current != null)
                {
                    stack.Push(current);
                    current = current.left;
                }
            }
            current = stack.Pop();
            // Node
            visitCount++;
            if (visitCount == k)
            {
                return current.val;
            }
            // RIGHT
            current = current.right;

        }
        // not found
        return -1;
    }

    public static int GetMinimumDifference(TreeNode root)
    {
        TreeNode prev = null;
        TreeNode curr = root;
        int globalMin = Int32.MaxValue;
        Stack<TreeNode> stack = new Stack<TreeNode>();

        while (curr != null || stack.Count > 0)
        {
            // Left
            while (curr != null)
            {
                stack.Push(curr);
                curr = curr.left;
            }
            curr = stack.Pop();
            // Visit
            if (prev != null)
            {
                int diff = Math.Abs(curr.val - prev.val);
                if (diff < globalMin) { globalMin = diff; }
            }
            prev = curr;
            curr = curr.right;
        }
        return globalMin;
    }


    public static IList<IList<int>> ZigzagLevelOrder(TreeNode root)
    {
        if (root == null) { return new List<IList<int>>(); }
        bool reverse = false;
        IList<IList<int>> list = new List<IList<int>>();
        Queue<TreeNode> q = new Queue<TreeNode>();
        q.Enqueue(root);
        while (q.Count > 0)
        {
            int levelCount = q.Count;
            List<int> sublist = new List<int>();
            for (int i = 0; i < levelCount; i++)
            {
                TreeNode node = q.Dequeue();
                sublist.Add(node.val);
                if (node.left != null) { q.Enqueue(node.left); }
                if (node.right != null) { q.Enqueue(node.right); }
            }
            if (reverse)
            {
                sublist.Reverse();
            }
            reverse = !reverse;
            list.Add(sublist);
        }
        return list;
    }



    public static IList<IList<int>> LevelOrder(TreeNode root)
    {
        if (root == null)
        {
            return new List<IList<int>>();
        }

        Queue<TreeNode> q = new Queue<TreeNode>();
        IList<IList<int>> levels = new List<IList<int>>();

        q.Enqueue(root);

        while (q.Count > 0)
        {
            IList<int> level = new List<int>();
            int levelCount = q.Count;

            for (int i = 0; i < levelCount; i++)
            {
                TreeNode node = q.Dequeue();
                level.Add(node.val);

                if (node.left != null) { q.Enqueue(node.left); }
                if (node.right != null) { q.Enqueue(node.right); }
            }
            levels.Add(level);
        }
        return levels;
    }



    public static IList<double> AverageOfLevels(TreeNode root)
    {
        if (root == null)
        {
            return null;
        }
        Queue<TreeNode> q = new Queue<TreeNode>();
        IList<double> avg = new List<double>();

        q.Enqueue(root);

        while (q.Count > 0)
        {
            int nodesInLevel = q.Count;
            double sum = 0;
            for (int i = 0; i < nodesInLevel; i++)
            {
                TreeNode node = q.Dequeue();
                sum += node.val;

                if (node.left != null) { q.Enqueue(node.left); }
                if (node.right != null) { q.Enqueue(node.right); }
            }
            avg.Add(sum / nodesInLevel);
        }
        return avg;
    }

    public static TreeNode LowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q)
    {
        if (root == null)
        {
            return null;
        }

        // This node matches either p or q
        if (root == p || root == q)
        {
            return root;
        }

        TreeNode left = LowestCommonAncestor(root.left, p, q);
        TreeNode right = LowestCommonAncestor(root.right, p, q);

        // both left and right subtrees have p and q
        if (left != null && right != null)
        {
            return root;
        }

        return left != null ? left : right;

    }



    public static int CountNodes(TreeNode root)
    {
        if (root == null)
        {
            return 0;
        }
        return CountNodes(root.left) + CountNodes(root.right) + 1;
    }

    private static int globalMax = int.MinValue;

    public static int MaxPathSum(TreeNode root)
    {
        MaxGain(root);
        return globalMax;
    }

    private static int MaxGain(TreeNode node)
    {
        if (node == null) return 0;

        int leftGain = Math.Max(MaxGain(node.left), 0);
        int rightGain = Math.Max(MaxGain(node.right), 0);

        int localMax = node.val + leftGain + rightGain;

        globalMax = Math.Max(globalMax, localMax);

        return node.val + Math.Max(leftGain, rightGain);
    }



    public static int SumNumbers(TreeNode root)
    {
        if (root == null) { return 0; }
        if (root.left == null && root.right == null) { return root.val; }

        return SumHelper(root.left, root.val) + SumHelper(root.right, root.val);
    }

    private static int SumHelper(TreeNode root, int runningSum)
    {
        if (root == null)
        {
            return 0;
        }
        runningSum = runningSum * 10 + root.val;
        // we are at the leat node, time to exit
        if (root.left == null && root.right == null)
        {
            return runningSum;
        }
        return SumHelper(root.left, runningSum) +
        SumHelper(root.right, runningSum);
    }

    public static bool HasPathSum(TreeNode root, int targetSum)
    {
        if (root == null) { return false; }

        int currentVal = root.val;
        targetSum = targetSum - currentVal;

        // leaf node
        if (root.left == null && root.right == null)
        {
            return targetSum == 0;
        }

        return HasPathSum(root.left, targetSum) || HasPathSum(root.right, targetSum);
    }


    public static void Flatten(TreeNode root)
    {
        TreeNode prev = null;
        FlattenPreorder(root, ref prev);
    }

    private static void FlattenPreorder(TreeNode node, ref TreeNode prev)
    {
        if (node == null) return;

        if (prev != null)
        {
            prev.right = node;
            prev.left = null; // clear left pointer
        }

        prev = node;

        TreeNode left = node.left;
        TreeNode right = node.right;

        FlattenPreorder(left, ref prev);
        FlattenPreorder(right, ref prev);
    }

    public static NodeConnect Connect(NodeConnect root)
    {
        if (root == null) { return null; }

        Queue<NodeConnect> elements = new Queue<NodeConnect>();
        elements.Enqueue(root);

        while (elements.Count > 0)
        {
            int levelSize = elements.Count;
            NodeConnect prev = null;

            for (int i = 0; i < levelSize; i++)
            {
                NodeConnect curr = elements.Dequeue();
                if (prev != null) { prev.next = curr; }
                prev = curr;

                if (curr.left != null) { elements.Enqueue(curr.left); }
                if (curr.right != null) { elements.Enqueue(curr.right); }
            }
            prev.next = null;
        }
        return root;
    }

    public static TreeNode BuildTree106(int[] inorder, int[] postorder)
    {

        Dictionary<int, int> inOrderlookup = new Dictionary<int, int>();

        for (int i = 0; i < inorder.Length; i++)
        {
            inOrderlookup.Add(inorder[i], i);
        }

        return BuildSubtree106(postorder.Length - 1, 0, inorder.Length - 1, inorder, postorder, inOrderlookup);

    }

    private static TreeNode BuildSubtree106(int postStart, int leftInIndex, int rightInIndex, int[] inorder, int[] postorder, Dictionary<int, int> inOrderlookup)
    {

        if (postStart < 0)
        {
            return null;
        }

        if (leftInIndex > rightInIndex)
        {
            return null;
        }

        TreeNode root = new TreeNode(postorder[postStart]);

        int inOrderPartitionValue = inOrderlookup[postorder[postStart]];
        int inOrderLeftSubtreeSize = inOrderPartitionValue - leftInIndex;
        int inOrderRightSubtreeSize = rightInIndex - inOrderPartitionValue;


        root.left = BuildSubtree106(postStart - inOrderRightSubtreeSize - 1, leftInIndex, inOrderPartitionValue - 1, inorder, postorder, inOrderlookup);
        root.right = BuildSubtree106(postStart - 1, inOrderPartitionValue + 1, rightInIndex, inorder, postorder, inOrderlookup);

        return root;

    }

    public static TreeNode BuildTree(int[] preorder, int[] inorder)
    {
        // Build value-to-index map for inorder traversal
        Dictionary<int, int> inorderIndexMap = new();
        for (int i = 0; i < inorder.Length; i++)
        {
            inorderIndexMap[inorder[i]] = i;
        }

        return BuildSubtree(0, 0, inorder.Length - 1, preorder, inorderIndexMap);
    }

    private static TreeNode BuildSubtree(int preStart, int inStart, int inEnd, int[] preorder, Dictionary<int, int> inorderIndexMap)
    {
        if (preStart >= preorder.Length || inStart > inEnd)
            return null;

        int rootVal = preorder[preStart];
        TreeNode root = new TreeNode(rootVal);

        int inIndex = inorderIndexMap[rootVal];
        int leftSubtreeSize = inIndex - inStart;

        root.left = BuildSubtree(preStart + 1, inStart, inIndex - 1, preorder, inorderIndexMap);
        root.right = BuildSubtree(preStart + 1 + leftSubtreeSize, inIndex + 1, inEnd, preorder, inorderIndexMap);

        return root;
    }



    public static bool IsSymmetric(TreeNode root)
    {
        if (root == null)
        {
            return true;
        }
        return IsMirror(root.left, root.right);
    }

    public static bool IsMirror(TreeNode left, TreeNode right)
    {
        if (left == null && right == null)
        {
            return true;
        }
        if (left == null || right == null)
        {
            return false;
        }
        return (left.val == right.val) && IsMirror(left.left, right.right) && IsMirror(left.right, right.left);
    }


    public static TreeNode InvertTree(TreeNode root)
    {
        if (root == null)
        {
            return null;
        }

        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;

        InvertTree(root.left);
        InvertTree(root.right);

        return root;
    }


    public static bool IsSameTree(TreeNode p, TreeNode q)
    {
        if (p == null && q == null)
        {
            return true;
        }

        if (p == null || q == null)
        {
            return false;
        }

        return p.val == q.val && IsSameTree(p.left, q.left) && IsSameTree(p.right, q.right);

    }


    public static int MaxDepth(TreeNode root)
    {
        if (root == null) return 0;
        return recursiveHelper(root, 1);
    }

    private static int recursiveHelper(TreeNode root, int currentDepth)
    {
        return Math.Max((root.left == null) ? currentDepth : recursiveHelper(root.left, currentDepth + 1),
                         (root.right == null) ? currentDepth : recursiveHelper(root.right, currentDepth + 1));

    }

    public static ListNode Partition(ListNode head, int x)
    {
        ListNode leftDummy = new ListNode(0);
        ListNode rightDummy = new ListNode(0);
        ListNode leftTail = leftDummy, rightTail = rightDummy;

        while (head != null)
        {
            if (head.val < x)
            {
                leftTail.next = head;
                leftTail = leftTail.next;
            }
            else
            {
                rightTail.next = head;
                rightTail = rightTail.next;
            }
            ListNode nextNode = head.next;
            head.next = null;
            head = nextNode;
        }

        leftTail.next = rightDummy.next;

        return leftDummy.next;
    }

    public static ListNode RotateRight(ListNode head, int k)
    {
        if (head == null || head.next == null || k == 0) return head;

        ListNode dummy = head;
        int count = 0;

        while (dummy != null)
        {
            count++;
            dummy = dummy.next;
        }

        dummy = head;
        // [ 1, 2, 3, 4, 5]
        //  D,H

        int remainder = k % count; // for big values of k
                                   //                     (5   - 2 - 1 ) = 2
                                   // disconnect  node  (count - k - 1) from the end, point the tail to head

        if (remainder == 0) return head;


        int fastForward = count - remainder - 1;

        while (fastForward > 0)
        {
            dummy = dummy.next;
            fastForward--;
        }

        // [ 1, 2, 3, 4, 5]
        //   H     D
        ListNode tail = dummy;
        ListNode newHead = dummy.next;

        tail.next = null;

        // [ 1, 2, 3 |  4, 5]
        //   H    T    NH,D  

        dummy = newHead;

        while (dummy.next != null)
        {
            dummy = dummy.next;
        }

        // [ 1, 2,  3 | 4, 5]
        //   H     T     NH D  
        dummy.next = head;

        return newHead;
    }


    public static ListNode DeleteDuplicates(ListNode head)
    {
        ListNode dummy = head;
        ListNode prev = null;

        while (dummy != null)
        {
            bool isDupe = false;
            while (dummy.next != null && dummy.val == dummy.next.val)
            {
                dummy.next = dummy.next.next;
                isDupe = true;
            }

            if (isDupe)
            {
                // Skip the entire dupe node
                if (prev == null)
                {
                    head = dummy.next;
                }
                else
                {
                    prev.next = dummy.next;
                }
                dummy = dummy.next;
            }
            else
            {
                prev = dummy;
                dummy = dummy.next;
            }


        }

        return head;
    }


    public static ListNode ReverseKGroup(ListNode head, int k)
    {
        if (head == null || k <= 1) return head;

        int count = 0;
        ListNode ptr = head;

        // Count total nodes
        while (ptr != null)
        {
            count++;
            ptr = ptr.next;
        }

        int left = 1;
        int right = k;

        while (right <= count)
        {
            head = ReverseBetweenEfficient(head, left, right);
            left += k;
            right += k;
        }

        return head;
    }


    public static ListNode ReverseBetweenEfficient(ListNode head, int left, int right)
    {
        if (head == null || left == right) return head;

        ListNode dummy = new ListNode(0, head);
        ListNode prev = dummy;

        // Move `prev` to the node just before the left-th node
        for (int i = 1; i < left; i++)
        {
            prev = prev.next;
        }

        // Start reversing from `current`
        ListNode current = prev.next;
        ListNode next = null;

        // Reverse nodes between left and right
        for (int i = 0; i < right - left; i++)
        {
            next = current.next;
            current.next = next.next;
            next.next = prev.next;
            prev.next = next;
        }

        return dummy.next;
    }

    public static ListNode ReverseBetween(ListNode head, int left, int right)
    {
        if (head == null || left == right) return head;

        Stack<ListNode> stack = new Stack<ListNode>();
        ListNode dummy = new ListNode(0, head);
        ListNode prev = dummy;


        for (int i = 1; i < left; i++)
        {
            prev = prev.next;
        }

        ListNode current = prev.next;
        for (int i = 0; i <= right - left; i++)
        {
            stack.Push(current);
            current = current.next;
        }

        // Re-link reversed nodes
        ListNode tail = current; // node after the reversed sublist
        while (stack.Count > 0)
        {
            prev.next = stack.Pop();
            prev = prev.next;
        }

        prev.next = tail;

        return dummy.next;
    }


    public static Node CopyRandomList(Node head)
    {
        if (head == null) return null;

        Dictionary<Node, Node> references = new Dictionary<Node, Node>();

        Node oldNode = head;

        // First pass: create new nodes and store them in the map
        while (oldNode != null)
        {
            references[oldNode] = new Node(oldNode.val);
            oldNode = oldNode.next;
        }

        oldNode = head;
        // Second pass: assign next and random pointers
        while (oldNode != null)
        {
            references[oldNode].next = oldNode.next != null ? references[oldNode.next] : null;
            references[oldNode].random = oldNode.random != null ? references[oldNode.random] : null;
            oldNode = oldNode.next;
        }

        return references[head];
    }

    public static ListNode MergeTwoLists(ListNode list1, ListNode list2)
    {
        ListNode dummy = new ListNode();
        ListNode tail = dummy;

        while (list1 != null && list2 != null)
        {
            if (list1.val <= list2.val)
            {
                tail.next = list1;
                list1 = list1.next;
            }
            else
            {
                tail.next = list2;
                list2 = list2.next;
            }
            tail = tail.next;
        }

        tail.next = list1 ?? list2;

        return dummy.next;
    }

    public static ListNode AddTwoNumbers(ListNode l1, ListNode l2)
    {

        ListNode dummyHead = new ListNode(0);
        ListNode current = dummyHead;
        int carry = 0;

        while (l1 != null || l2 != null || carry != 0)
        {
            int val1 = l1 != null ? l1.val : 0;
            int val2 = l2 != null ? l2.val : 0;

            int sum = val1 + val2 + carry;
            carry = sum / 10;

            current.next = new ListNode(sum % 10);
            current = current.next;

            if (l1 != null) l1 = l1.next;
            if (l2 != null) l2 = l2.next;
        }

        return dummyHead.next;
    }

    public static bool HasCycle(ListNode head)
    {

        if (head == null || head.next == null)
        {
            return false;
        }

        ListNode slow = head;
        ListNode fast = head;

        while (fast != null && fast.next != null)
        {
            slow = slow.next;
            fast = fast.next.next;

            if (slow == fast)
            {
                return true;
            }
        }

        return false;

    }

    public static int Calculate(string s)
    {
        Stack<int> stack = new Stack<int>();
        int result = 0;
        int sign = 1;
        int num = 0;

        for (int i = 0; i < s.Length; i++)
        {
            char c = s[i];

            if (char.IsDigit(c))
            {
                num = num * 10 + (c - '0');
            }
            else if (c == '+')
            {
                result += sign * num;
                num = 0;
                sign = 1;
            }
            else if (c == '-')
            {
                result += sign * num;
                num = 0;
                sign = -1;
            }
            else if (c == '(')
            {
                stack.Push(result);
                stack.Push(sign);
                result = 0;
                sign = 1;
            }
            else if (c == ')')
            {
                result += sign * num;
                num = 0;
                result *= stack.Pop(); // sign
                result += stack.Pop(); // previous result
            }
            // else skip whitespace
        }

        result += sign * num;
        return result;
    }


    public static int EvalRPN(string[] tokens)
    {

        // if digit, then push into stack
        // if operand, then popx2 
        // and push result

        Stack<int> calculator = new Stack<int>();
        for (int i = 0; i < tokens.Length; i++)
        {

            string token = tokens[i];

            if (token == "+" ||
                token == "-" ||
                token == "*" ||
                token == "/")
            {

                int rhs = calculator.Pop();
                int lhs = calculator.Pop();
                int result = 0;

                switch (token)
                {
                    case "+":
                        result = lhs + rhs;
                        break;
                    case "-":
                        result = lhs - rhs;
                        break;
                    case "/":
                        result = lhs / rhs;
                        break;
                    case "*":
                        result = lhs * rhs;
                        break;
                    default:
                        throw new ArgumentException("Invalid operator");
                }
                calculator.Push(result);
            }
            else
            {
                calculator.Push(int.Parse(token));

            }
        }
        return (int)calculator.Pop();
    }

    public static class MinStack
    {
        static Stack<int[]> minStack = new Stack<int[]>();
        public static void Push(int val)
        {
            // each time I push in, I also push current minimum
            if (minStack.Count == 0)
            {
                minStack.Push(new int[] { val, val });
            }
            else
            {
                minStack.Push(new int[] { val, Math.Min(minStack.Peek()[1], val) });
            }
        }

        public static void Pop()
        {
            minStack.Pop();
        }

        public static int Top()
        {
            return minStack.Peek()[0];
        }

        public static int GetMin()
        {
            return minStack.Peek()[1];
        }
    }



    public static string SimplifyPath(string path)
    {
        Stack<string> stack = new Stack<string>();
        string[] parts = path.Split('/');

        foreach (string part in parts)
        {
            if (string.IsNullOrEmpty(part) || part == ".")
            {
                continue;
            }
            else if (part == "..")
            {
                if (stack.Count > 0)
                {
                    stack.Pop();
                }
            }
            else
            {
                stack.Push(part);
            }
        }

        if (stack.Count == 0) return "/";

        var result = new StringBuilder();
        foreach (var dir in stack.Reverse())
        {
            result.Append('/').Append(dir);
        }

        return result.ToString();
    }


    public static bool IsValid(string s)
    {
        Stack<char> myStack = new Stack<char>();

        for (int i = 0; i < s.Length; i++)
        {
            char brace = s[i];
            if (brace == '[' || brace == '(' || brace == '{')
            {
                myStack.Push(brace);
                continue;
            }

            if (myStack.Count == 0) return false;
            char leftBrace = myStack.Pop();

            if (leftBrace == '(' && brace == ')' ||
            leftBrace == '[' && brace == ']'
            || leftBrace == '{' && brace == '}')
            {
                continue;
            }
            return false;
        }

        return myStack.Count == 0;
    }

    public static int FindMinArrowShots(int[][] points)
    {
        if (points.Length == 0) return 0;

        // sort by right edge (end) 
        Array.Sort(points, (a, b) => a[1].CompareTo(b[1]));

        int i = 0;
        int minArrows = 0;

        // [1---6]     1st arrow
        //   [2-----8] 1st arrow
        //        [7------12]     2nd arrow   
        //             [10------16]  2nd arrow
        while (i < points.Length)
        {
            // shoot arrow at the end of current balloon
            int arrowPos = points[i][1];
            minArrows++;

            // skip all balloons that this arrow can burst
            while (i < points.Length && points[i][0] <= arrowPos)
            {
                i++;
            }
        }

        return minArrows;
    }

    public static int[][] Insert(int[][] intervals, int[] newInterval)
    {
        int i = 0;
        List<int[]> newIntervals = new List<int[]>();

        int newLeft = newInterval[0];
        int newRight = newInterval[1];

        // add all intervals that come before newInterval
        while (i < intervals.Length && intervals[i][1] < newLeft)
        {
            newIntervals.Add(intervals[i]);
            i++;
        }

        // merge overlapping intervals
        while (i < intervals.Length && intervals[i][0] <= newRight)
        {
            newLeft = Math.Min(newLeft, intervals[i][0]);
            newRight = Math.Max(newRight, intervals[i][1]);
            i++;
        }

        newIntervals.Add(new int[] { newLeft, newRight });

        // add remaining intervals after the newInterval
        while (i < intervals.Length)
        {
            newIntervals.Add(intervals[i]);
            i++;
        }
        return newIntervals.ToArray();
    }

    public static int[][] Merge(int[][] intervals)
    {

        if (intervals == null || intervals.Length == 0)
            return new int[0][];

        // Sort by a
        Array.Sort(intervals, (a, b) => a[0].CompareTo(b[0]));

        List<int[]> newIntervals = new List<int[]>();

        int i = 0;
        while (i < intervals.Length)
        {
            int left = intervals[i][0];
            int right = intervals[i][1];

            // Merge all overlapping intervals
            while (i < intervals.Length - 1 && intervals[i + 1][0] <= right)
            {
                i++;
                right = Math.Max(right, intervals[i][1]);
            }

            newIntervals.Add(new int[] { left, right });
            i++;
        }

        return newIntervals.ToArray();

    }

    public static IList<string> SummaryRangesSimplified(int[] nums)
    {
        List<string> summaryRange = new List<string>();

        int i = 0;
        while (i < nums.Length)
        {

            int start = i;
            while (i < nums.Length - 1 && nums[i] + 1 == nums[i + 1])
            {
                i++;
            }
            int end = i;

            if (start == end)
            {
                summaryRange.Add($"{nums[start]}");
            }
            else
            {
                summaryRange.Add($"{nums[start]}->{nums[end]}");
            }

            i++;

        }

        return summaryRange;

    }


    public static IList<string> SummaryRanges(int[] nums)
    {
        List<string> summaryRange = new List<string>();

        for (int i = 0; i < nums.Length; i++)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(nums[i]);
            if (i >= nums.Length - 1 || nums[i] + 1 != nums[i + 1])
            {
                summaryRange.Add(sb.ToString());
                continue;
            }

            while (i < nums.Length - 1 && nums[i] + 1 == nums[i + 1])
            {
                i++;
            }
            sb.Append("->");
            sb.Append(nums[i]);

            summaryRange.Add(sb.ToString());
        }
        return summaryRange;
    }


    public static int LongestConsecutive(int[] nums)
    {

        HashSet<int> hashNums = new HashSet<int>(nums);
        int longest = 0;

        foreach (var num in hashNums)
        {
            if (hashNums.Contains(num - 1))
            {
                // not at the start of the set, skip
                continue;
            }

            int length = 1;
            int currentNum = num;

            while (hashNums.Contains(currentNum + 1))
            {
                currentNum++;
                length++;
            }

            longest = Math.Max(longest, length);
        }

        return longest;
    }

    public static bool ContainsNearbyDuplicate(int[] nums, int k)
    {
        // num,index
        Dictionary<int, int> seen = new Dictionary<int, int>();
        for (int i = 0; i < nums.Length; i++)
        {
            if (seen.ContainsKey(nums[i]) && (i - seen[nums[i]] <= k))
            {
                return true;
            }
            seen[nums[i]] = i; // update with the latest index
        }
        return false;
    }

    public static int[] TwoSumClean(int[] nums, int target)
    {

        Dictionary<int, int> indcies = new Dictionary<int, int>();
        // Map all the values into a dictionary

        for (int i = 0; i < nums.Length; i++)
        {
            int compliment = target - nums[i];
            if (indcies.ContainsKey(compliment))
            {
                return new int[] { i, indcies[compliment] };
            }

            if (!indcies.ContainsKey(nums[i]))
            {
                indcies.Add(nums[i], i);
            }
        }
        return new int[] { };
    }

    public static int[] TwoSum(int[] nums, int target)
    {

        Dictionary<int, int> indcies = new Dictionary<int, int>();
        // Map all the values into a dictionary
        for (int i = 0; i < nums.Length; i++)
        {
            if (!indcies.ContainsKey(nums[i]))
            {
                indcies.Add(nums[i], i);
            }
        }

        for (int i = 0; i < nums.Length; i++)
        {
            int compliment = target - nums[i];
            if (indcies.ContainsKey(compliment) && (indcies[compliment] != i))
            {
                return new int[] { i, indcies[compliment] };
            }
        }
        return new int[] { };
    }


    public static IList<IList<string>> GroupAnagramsEfficient(string[] strs)
    {

        Dictionary<string, List<string>> map = new Dictionary<string, List<string>>();

        foreach (string s in strs)
        {
            char[] chars = s.ToCharArray();
            Array.Sort(chars);
            string key = new string(chars); // key is sorted chars

            if (!map.ContainsKey(key))
            {
                map[key] = new List<string>();
            }
            map[key].Add(s); // value is a list, appending each the candidate string that matches
        }

        return new List<IList<string>>(map.Values);
    }



    public static IList<IList<string>> GroupAnagrams(string[] strs)
    {

        IList<IList<string>> listOfLists = new List<IList<string>>();
        Dictionary<int, string> map = new Dictionary<int, string>();

        // This helps to keep track of everything
        // rather than removing over the iterator
        for (int i = 0; i < strs.Length; i++)
        {
            map[i] = strs[i];
        }

        for (int i = 0; i < strs.Length; i++)
        {
            if (!map.ContainsKey(i))
            {
                continue;
            }

            string candidate = strs[i];
            List<string> currentList = new List<string>();
            currentList.Add(candidate);
            map.Remove(i);

            // check all that is remaining in the list
            for (int j = i; j < strs.Length; j++)
            {
                if (!map.ContainsKey(j))
                {
                    continue;
                }

                string nextCandidate = strs[j];
                if (IsAnagram(candidate, nextCandidate))
                {
                    currentList.Add(nextCandidate);
                    map.Remove(j);
                }
            }
            listOfLists.Add(currentList);
        }

        return listOfLists;
    }



    public static bool IsAnagramEfficient(string s, string t)
    {
        if (s.Length != t.Length) return false;
        int[] f = new int[26];
        foreach (char c in s) { ++f[c - 'a']; }
        foreach (char c in t)
        {
            if (f[c - 'a'] > 0) --f[c - 'a'];
            else
            {
                return false;
            }
        }
        return true;
    }

    public static bool IsAnagram(string s, string t)
    {

        //Need to check cases like
        // s = "aabb" and "ab" 
        if (s.Length != t.Length) return false;

        Dictionary<char, int> charCount = new Dictionary<char, int>();
        for (int i = 0; i < s.Length; i++)
        {
            char c = s[i];
            if (!charCount.ContainsKey(c)) { charCount[c] = 0; }
            charCount[c]++;
        }

        for (int i = 0; i < t.Length; i++)
        {
            char c = t[i];
            if (!charCount.ContainsKey(c) || charCount[c] == 0)
            {
                return false;
            }
            charCount[c]--;
        }
        return true;
    }

    public static bool WordPattern(string pattern, string s)
    {
        Dictionary<char, string> patternToString = new Dictionary<char, string>();
        Dictionary<string, char> stringToPattern = new Dictionary<string, char>();
        string[] tokens = s.Split(' ');

        if (pattern.Length != tokens.Length)
        {
            return false;
        }

        for (int i = 0; i < pattern.Length; i++)
        {

            char c = pattern[i];
            string token = tokens[i];

            if (patternToString.ContainsKey(c))
            {
                if (patternToString[c] != token)
                {
                    return false;
                }

            }
            else
            {
                patternToString[c] = token;
            }

            if (stringToPattern.ContainsKey(token))
            {
                if (stringToPattern[token] != c)
                {
                    return false;
                }
            }
            else
            {
                stringToPattern[token] = c;
            }
        }
        return true;
    }

    public static bool CanConstruct(string ransomNote, string magazine)
    {
        Dictionary<char, int> mag = new Dictionary<char, int>();
        foreach (char c in magazine)
        {
            if (mag.ContainsKey(c))
            {
                mag[c] = mag[c] + 1;
            }
            else
                mag.Add(c, 1);
        }
        foreach (char c in ransomNote)
        {
            if (!mag.ContainsKey(c)) return false;
            else
            {
                int count = mag[c];
                if (count == 0) return false;
                mag[c] = mag[c] - 1;
            }
        }
        return true;
    }


    public static void GameOfLifeInPlace(int[][] board)
    {
        int rows = board.Length;
        int cols = board[0].Length;

        int[] dx = { -1, -1, -1, 0, 0, 1, 1, 1 };
        int[] dy = { -1, 0, 1, -1, 1, -1, 0, 1 };

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                int liveNeighbors = 0;

                for (int d = 0; d < 8; d++)
                {
                    int ni = i + dx[d];
                    int nj = j + dy[d];
                    if (ni >= 0 && ni < rows && nj >= 0 && nj < cols)
                    {
                        liveNeighbors += board[ni][nj] & 1; // current state
                    }
                }

                // Apply Game of Life rules using current state (LSB) and write new state to 2nd bit
                if ((board[i][j] & 1) == 1)
                {
                    if (liveNeighbors == 2 || liveNeighbors == 3)
                    {
                        board[i][j] |= 2; // Set second bit to 1 → cell stays alive
                    }
                }
                else
                {
                    if (liveNeighbors == 3)
                    {
                        board[i][j] |= 2; // Set second bit to 1 → cell becomes alive
                    }
                }
            }
        }

        // Final pass: shift each cell right 1 bit to update to new state
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                board[i][j] >>= 1;
            }
        }
    }

    public static void GameOfLife(int[][] board)
    {

        int rows = board.Length;
        int columns = board[0].Length;

        int[][] nextState = new int[rows][];
        for (int i = 0; i < rows; i++)
        {
            nextState[i] = new int[columns];
        }

        for (int i = 0; i < board.Length; i++)
        {
            for (int j = 0; j < board[i].Length; j++)
            {
                nextState[i][j] = nextVal(board, i, j);
            }
        }

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                board[i][j] = nextState[i][j];
            }
        }

    }

    // Checks the conditions of the surrounding squares
    private static int nextVal(int[][] board, int i, int j)
    {
        int rows = board.Length;
        int columns = board[0].Length;
        int onesCount = 0;
        int cellValue = board[i][j];

        bool canGoTop = i > 0;
        bool canGoBottom = i < rows - 1;
        bool canGoRight = j < columns - 1;
        bool canGoLeft = j > 0;


        // top: i-1, j
        if (canGoTop)
        {
            onesCount += board[i - 1][j];

            // top left: i-1, j-1
            if (canGoLeft)
            {
                onesCount += board[i - 1][j - 1];
            }

            // top right: i-1, j+1 
            if (canGoRight)
            {
                onesCount += board[i - 1][j + 1];
            }

        }

        // bottom: i+1, j
        if (canGoBottom)
        {
            onesCount += board[i + 1][j];

            // bottom left: i+1, j-1
            if (canGoLeft)
            {
                onesCount += board[i + 1][j - 1];
            }

            // bottom right: i+1, j+1
            if (canGoRight)
            {
                onesCount += board[i + 1][j + 1];
            }
        }

        // left: i, j-1
        if (canGoLeft)
        {
            onesCount += board[i][j - 1];
        }

        // right: i, j+1
        if (canGoRight)
        {
            onesCount += board[i][j + 1];
        }

        if (cellValue == 0 && onesCount == 3)
        {
            return 1;
        }

        if (cellValue == 1 && (onesCount < 2 || onesCount > 3))
        {
            return 0;
        }

        if (cellValue == 1 && (onesCount == 2 || onesCount == 3))
        {
            return 1;
        }

        return board[i][j];


    }


    public static void SetZeroesNoSpace(int[][] matrix)
    {
        int rows = matrix.Length;
        int cols = matrix[0].Length;

        bool firstRowZero = false;
        bool firstColZero = false;

        // Check if first row has any zeros
        for (int j = 0; j < cols; j++)
        {
            if (matrix[0][j] == 0)
            {
                firstRowZero = true;
                break;
            }
        }

        // Check if first column has any zeros
        for (int i = 0; i < rows; i++)
        {
            if (matrix[i][0] == 0)
            {
                firstColZero = true;
                break;
            }
        }

        // Use first row and column as markers
        for (int i = 1; i < rows; i++)
        {
            for (int j = 1; j < cols; j++)
            {
                if (matrix[i][j] == 0)
                {
                    matrix[i][0] = 0;
                    matrix[0][j] = 0;
                }
            }
        }

        // Zero out cells based on markers
        for (int i = 1; i < rows; i++)
        {
            for (int j = 1; j < cols; j++)
            {
                if (matrix[i][0] == 0 || matrix[0][j] == 0)
                {
                    matrix[i][j] = 0;
                }
            }
        }

        // Zero out first row if needed
        if (firstRowZero)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[0][j] = 0;
            }
        }

        // Zero out first column if needed
        if (firstColZero)
        {
            for (int i = 0; i < rows; i++)
            {
                matrix[i][0] = 0;
            }
        }
    }


    public static void SetZeroes(int[][] matrix)
    {

        int rows = matrix.Length;
        int columns = matrix[0].Length;

        BitArray bitwiseRows = new BitArray(rows);
        BitArray bitwiseColumns = new BitArray(columns);

        // First need to find 0's otherwise we risk setting them
        // unnecessarily


        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                if (matrix[i][j] == 0)
                {
                    bitwiseRows[i] = true;
                    bitwiseColumns[j] = true;
                }
            }
        }

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                if (bitwiseRows[i])
                {
                    matrix[i][j] = 0;
                    continue;
                }

                if (bitwiseColumns[j])
                {
                    matrix[i][j] = 0;
                    continue;
                }
            }
        }

    }




    // Transpose
    // 1 4 7 -> 7 4 1
    // 2 5 8 -> 8 5 2
    // 3 6 9 -> 9 6 3

    // Reverse
    // 7 4 1 -> 1 4 7
    // 8 5 2 -> 2 5 8 
    // 9 6 3 -> 3 6 9 
    public static void Rotate(int[][] matrix)
    {
        int n = matrix.Length;

        // Transpose
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }

        // Reverse
        for (int i = 0; i < n; i++)
        {
            Array.Reverse(matrix[i]);
        }
    }


    //[ [1,2,3],
    //  [4,5,6],
    //  [7,8,9]  ]

    // maxTop - Top range: matrix[]

    // Reduce top range by one
    // Right -> reduce right range by one -> 
    // Down -> reduce down range by one
    // Left -> reduce left range by one
    // Top -> reduce top range by one
    //
    // Repeat

    // flag right until end of column at row


    //(0,0) -> (0,1) -> (0,2)
    //(1,2) -> (2,3) 
    //->         
    public static IList<int> SpiralOrder(int[][] matrix)
    {

        int maxRight = matrix[0].Length - 1;
        int maxLeft = 0;
        int maxTop = 0;
        int maxBottom = matrix.Length - 1;
        List<int> results = new List<int>();

        while (maxTop <= maxBottom && maxLeft <= maxRight)
        {

            // Move Right
            for (int j = maxLeft; j <= maxRight; j++)
            {
                results.Add(matrix[maxTop][j]);
            }
            maxTop++;

            // Move Down
            for (int i = maxTop; i <= maxBottom; i++)
            {
                results.Add(matrix[i][maxRight]);
            }
            maxRight--;

            // Move Left
            if (maxTop <= maxBottom)
            {
                for (int j = maxRight; j >= maxLeft; j--)
                {
                    results.Add(matrix[maxBottom][j]);
                }
                maxBottom--;
            }

            // Move Up
            if (maxLeft <= maxRight)
            {
                for (int i = maxBottom; i >= maxTop; i--)
                {
                    results.Add(matrix[i][maxLeft]);
                }
                maxLeft++;
            }

        }


        return results;


    }

    public static bool IsValidSudoku(char[][] board)
    {
        // Check rows
        for (int i = 0; i < 9; i++)
        {
            if (!IsSubSolutionValid(board, i, i + 1, 0, 9))
                return false;
        }

        // Check columns
        for (int i = 0; i < 9; i++)
        {
            if (!IsSubSolutionValid(board, 0, 9, i, i + 1))
                return false;
        }

        // Check 3x3 subgrids
        for (int row = 0; row < 9; row += 3)
        {
            for (int col = 0; col < 9; col += 3)
            {
                if (!IsSubSolutionValid(board, row, row + 3, col, col + 3))
                    return false;
            }
        }

        return true;
    }

    private static bool IsSubSolutionValid(char[][] board, int startRow, int endRow, int startCol, int endCol)
    {
        HashSet<char> setOfNine = new HashSet<char>();

        for (int i = startRow; i < endRow; i++)
        {
            for (int j = startCol; j < endCol; j++)
            {
                char candidate = board[i][j];
                if (candidate != '.')
                {
                    if (setOfNine.Contains(candidate)) return false;
                    setOfNine.Add(candidate);
                }
            }
        }

        return true;
    }

    public static string MinWindow(string s, string t)
    {
        if (string.IsNullOrEmpty(s) || string.IsNullOrEmpty(t)) return "";

        Dictionary<char, int> tCount = new Dictionary<char, int>();
        foreach (char c in t)
        {
            if (!tCount.ContainsKey(c)) tCount[c] = 0;
            tCount[c]++;
        }

        Dictionary<char, int> window = new Dictionary<char, int>();
        int have = 0;
        int need = tCount.Count;
        int left = 0;

        int resLen = int.MaxValue;
        int resStart = 0;


        for (int right = 0; right < s.Length; right++)
        {
            char c = s[right];
            if (!window.ContainsKey(c)) window[c] = 0;
            window[c]++;

            if (tCount.ContainsKey(c) && window[c] == tCount[c])
                have++;


            while (have == need)
            {

                // find the smallest window 
                if (right - left + 1 < resLen)
                {
                    resStart = left;
                    resLen = right - left + 1;
                }

                char leftChar = s[left];
                window[leftChar]--;
                if (tCount.ContainsKey(leftChar) && window[leftChar] < tCount[leftChar])
                    have--;

                left++;
            }
        }

        return resLen == int.MaxValue ? "" : s.Substring(resStart, resLen);


    }


    public static IList<int> FindSubstring(string s, string[] words)
    {
        if (s == null || s.Length == 0 || words == null || words.Length == 0) return new List<int>();


        int wordLength = words[0].Length;
        int totalWords = words.Length;
        int totalLength = wordLength * totalWords;


        Dictionary<string, int> wordCount = new Dictionary<string, int>();

        foreach (string word in words)
        {
            if (!wordCount.ContainsKey(word)) wordCount[word] = 0;
            wordCount[word]++;
        }

        List<int> result = new List<int>();


        for (int i = 0; i < wordLength; i++)
        {
            int left = i;
            int count = 0;
            Dictionary<string, int> window = new Dictionary<string, int>();

            for (int j = i; j <= s.Length - wordLength; j += wordLength)
            {
                string w = s.Substring(j, wordLength);
                if (wordCount.ContainsKey(w))
                {
                    if (!window.ContainsKey(w)) window[w] = 0;
                    window[w]++;
                    count++;

                    // If word appears too many times, move left pointer
                    while (window[w] > wordCount[w])
                    {
                        string leftWord = s.Substring(left, wordLength);
                        window[leftWord]--;
                        left += wordLength;
                        count--;
                    }

                    if (count == totalWords)
                    {
                        result.Add(left);
                    }
                }
                else
                {
                    window.Clear();
                    count = 0;
                    left = j + wordLength;
                }
            }
        }

        return result;


    }

    public static int LengthOfLongestSubstring(string s)
    {

        // Uh oh, a dupe
        // "abcabcbb"
        //  L  R

        // Move the window until dupe isn't there
        // "abcabcbb"
        //   L R


        int l = 0;
        int maxLength = 0;
        HashSet<char> window = new HashSet<char>();

        for (int r = 0; r < s.Length; r++)
        {
            while (window.Contains(s[r]))
            {
                window.Remove(s[l]);
                l++;
            }

            window.Add(s[r]);
            maxLength = Math.Max(maxLength, r - l + 1);
        }

        return maxLength;


    }


    public static int MinSubArrayLen(int target, int[] nums)
    {

        // Add to sum Going Right with R
        // Subtract sum Going Right with L
        // Subtract Left while greater or equal
        // use the minimum that has the smallest diff between L and R
        // [2,3,1,2,4,3]
        // LR
        int sum = 0;
        int min = int.MaxValue;
        int L = 0;

        // fix and slide        
        for (int R = 0; R < nums.Length; R++)
        {
            sum += nums[R];


            while (sum >= target)
            {
                min = Math.Min(R - L + 1, min);
                sum -= nums[L];
                L++;
            }
        }


        return min == int.MaxValue ? 0 : min;
    }


    public static IList<IList<int>> ThreeSumPointer(int[] nums)
    {
        Array.Sort(nums);
        var result = new List<IList<int>>();

        for (int i = 0; i < nums.Length - 2; i++)
        {
            if (i > 0 && nums[i] == nums[i - 1])
                continue; // Skip duplicate `i`

            int left = i + 1;
            int right = nums.Length - 1;

            while (left < right)
            {
                int sum = nums[i] + nums[left] + nums[right];

                if (sum == 0)
                {
                    result.Add(new List<int> { nums[i], nums[left], nums[right] });

                    // Move left/right and skip duplicates
                    int leftVal = nums[left];
                    int rightVal = nums[right];

                    while (left < right && nums[left] == leftVal) left++;
                    while (left < right && nums[right] == rightVal) right--;
                }
                else if (sum < 0)
                {
                    left++;
                }
                else
                {
                    right--;
                }
            }
        }

        return result;
    }

    public static IList<IList<int>> ThreeSumHash(int[] nums)
    {
        var results = new HashSet<string>(); // Use a set to prevent duplicates
        var final = new List<IList<int>>();

        for (int i = 0; i < nums.Length; i++)
        {
            int fixedNum = nums[i];
            var seen = new HashSet<int>();

            for (int j = i + 1; j < nums.Length; j++)
            {
                int complement = -fixedNum - nums[j];

                if (seen.Contains(complement))
                {
                    var triplet = new List<int> { fixedNum, nums[j], complement };
                    triplet.Sort(); // Ensure triplets are always in same order for deduplication
                    string key = string.Join(",", triplet);
                    if (!results.Contains(key))
                    {
                        results.Add(key);
                        final.Add(triplet);
                    }
                }

                seen.Add(nums[j]);
            }
        }

        return final;
    }

    public static int MaxArea(int[] height)
    {
        int left = 0;
        int right = height.Length - 1;
        int maxVolume = 0;

        while (left < right)
        {
            int width = right - left;
            int minHeight = Math.Min(height[left], height[right]);
            int currentVol = minHeight * width;
            maxVolume = Math.Max(maxVolume, currentVol);

            if (height[left] < height[right])
            {
                left++;
            }
            else
            {
                right--;
            }
        }

        return maxVolume;
    }



    // This one only works because the input GUARANTEES a solution
    // AND it is GUARANTEED to be sorted.
    // Because of this constraint, you just keep moving inwards until
    // find the solution that is waiting for you.
    // 
    // Left inwards if it's too small, right inwards if it's too big.
    public static int[] TwoSumOld(int[] numbers, int target)
    {
        int left = 0;
        int right = numbers.Length - 1;

        while (left < right)
        {
            int sum = numbers[left] + numbers[right];
            if (sum == target)
            {
                return new int[] { left + 1, right + 1 };
            }
            if (sum < target)
            {
                left++;
            }
            else
            {
                right--;
            }
        }

        return new int[0];
    }


    public static bool IsSubsequence(string s, string t)
    {
        int schar = 0;
        int tchar = 0;

        while (schar < s.Length && tchar < t.Length)
        {
            if (s[schar] == t[tchar])
            {
                schar++;
            }
            tchar++;
        }

        return schar == s.Length;
    }


    public static bool IsPalindrome(string s)
    {

        if (s == null) return false;


        // sanitize input

        // This is probably better programming style
        // for clarity but this is frowned upon in Leet as I understand
        //
        //string sanitized = s.ToUpper();
        //Regex rgx = new Regex("[^a-zA-Z0-9]");
        //sanitized = rgx.Replace(sanitized, "");

        int left = 0;
        int right = s.Length - 1;


        while (left < right)
        {

            while (left < right && !char.IsLetterOrDigit(s[left])) left++;
            while (left < right && !char.IsLetterOrDigit(s[right])) right--;

            if (char.ToUpper(s[left]) != char.ToUpper(s[right]))
            {
                return false;
            }

            right--;
            left++;

        }

        return true;

    }



    public static IList<string> FizzBuzz(int n)
    {

        List<string> fizzBuzz = new List<string>();

        for (int i = 1; i < n + 1; i++)
        {
            if (i % 3 == 0 && i % 5 == 0)
            {
                fizzBuzz.Add("FizzBuzz");
                continue;
            }

            if (i % 3 == 0)
            {
                fizzBuzz.Add("Fizz");
                continue;
            }

            if (i % 5 == 0)
            {
                fizzBuzz.Add("Buzz");
                continue;
            }

            fizzBuzz.Add(i.ToString());
        }

        return fizzBuzz;

    }


    public static int StrStr(string haystack, string needle)
    {

        if (needle.Length == 0) return 0;

        for (int i = 0; i <= haystack.Length - needle.Length; i++)
        {
            if (haystack[i] == needle[0])
            {
                bool isMatch = true;
                for (int j = 1; j < needle.Length; j++)
                {
                    if (haystack[i + j] != needle[j])
                    {
                        isMatch = false;
                        break;
                    }
                }
                if (isMatch) return i;
            }
        }

        return -1;


    }

    static string ConvertPayPalIsHiringZigZag(string s, int numRows)
    {

        if (numRows == 1)
        {
            return s;
        }

        // PAYPALISHIRING , 3
        StringBuilder[] listOfStrings = new StringBuilder[numRows];
        for (int j = 0; j < numRows; j++)
        {
            listOfStrings[j] = new StringBuilder();
        }

        bool flipDirection = true;
        int zigZagIndex = 0;

        // Build Up
        //[PAHN]
        //[APLSIIG]
        //[YIR]
        for (int i = 0; i < s.Length; i++)
        {
            listOfStrings[zigZagIndex].Append(s[i]);

            // 0,1,2,1,0
            if (zigZagIndex == numRows - 1 || zigZagIndex == 0)
            {
                flipDirection = !flipDirection;
            }

            if (flipDirection)
            {
                zigZagIndex = zigZagIndex - 1;
            }
            else
            {
                zigZagIndex = zigZagIndex + 1;
            }

        }


        StringBuilder result = new StringBuilder();
        foreach (var sb in listOfStrings)
        {
            result.Append(sb);
        }

        return result.ToString();
    }


    static string ReverseWordsOptimized(string s)
    {
        // Convert string to char array for in-place modifications
        char[] charArray = s.ToCharArray();

        int start = 0;
        int end = charArray.Length - 1;

        while (start <= end && charArray[start] == ' ') start++; // Trim leading spaces
        while (end >= start && charArray[end] == ' ') end--;   // Trim trailing spaces

        // Step 1: Reverse the entire string
        Reverse(charArray, start, end);

        // Step 2: Reverse each word in-place
        int wordStart = start;
        for (int i = start; i <= end; i++)
        {
            if (charArray[i] == ' ' || i == end)
            {
                int wordEnd = (i == end) ? i : i - 1;
                Reverse(charArray, wordStart, wordEnd);
                wordStart = i + 1;
            }
        }

        // Step 3: Clean up extra spaces between words (by compacting in-place)
        int writeIndex = start;
        for (int readIndex = start; readIndex <= end; readIndex++)
        {
            // Skip extra spaces
            if (readIndex == start && charArray[readIndex] == ' ') continue;
            if (readIndex > start && charArray[readIndex] == ' ' && charArray[readIndex - 1] == ' ') continue;

            // Write valid characters to the result array
            charArray[writeIndex++] = charArray[readIndex];
        }

        // Final conversion of charArray back to string
        return new string(charArray, start, writeIndex - start);
    }

    private static void Reverse(char[] charArray, int start, int end)
    {
        while (start < end)
        {
            char temp = charArray[start];
            charArray[start] = charArray[end];
            charArray[end] = temp;
            start++;
            end--;
        }
    }


    static string ReverseWords(string s)
    {

        int i = 0;
        int j = 0;

        // trim leading whitespaces
        while (i < s.Length)
        {

            if (s[0] == ' ')
            {
                s = s.Substring(1);
            }
            else
            {
                break;
            }
            i++;
        }

        i = s.Length - 1;
        // trim leading whitespaces
        while (i > 0)
        {

            if (s[s.Length - 1] == ' ')
            {
                s = s.Substring(0, s.Length - 1);
            }
            else
            {
                break;
            }
            i--;
        }

        // trim multiple whitespaces
        i = 0;
        while (i < s.Length)
        {

            if (i + 1 < s.Length && s[i] == ' ' && s[i + 1] == ' ')
            {
                s = s.Substring(0, i) + s.Substring(i + 1);
                i--; // adjust not to overshoot
            }
            i++;
        }


        s = reverse(s, 0, s.Length);

        i = 0;

        while (i < s.Length)
        {

            if (s[i] == ' ')
            {
                // found word boundary
                s = reverse(s, j, i);
                j = i + 1;
                i = j;
            }
            else
            {
                i++;
            }
        }

        s = reverse(s, j, s.Length); // cannot omit the last word

        return s;
    }

    static string reverse(string input, int start, int end)
    {
        int left = start;
        int right = end - 1;

        char[] charArray = input.ToCharArray();

        while (left < right)
        {
            char placeholder = charArray[left];
            charArray[left] = charArray[right];
            charArray[right] = placeholder;
            left++;
            right--;

        }
        return new string(charArray);
    }


    static string LongestCommonPrefix(string[] strs)
    {

        string candidate = strs[0];

        int i = 1;
        while (i < strs.Length)
        {

            int maxMatch = 0;
            for (int j = 0; j < strs[i].Length; j++)
            {

                if (j < candidate.Length && strs[i][j] == candidate[j])
                {
                    maxMatch++;
                }
                else
                {
                    break;
                }
            }
            candidate = candidate.Substring(0, maxMatch);
            i++;
        }
        return candidate;
    }



    static int LengthOfLastWordNoStringOperation(string s)
    {

        int sum = 0;
        int i = s.Length - 1;

        // skip whitespace
        while (i >= 0 && s[i] == ' ') i--;

        while (i >= 0 && s[i] != ' ')
        {
            sum += 1;
            i--;
        }

        return sum;

    }


    static int LengthOfLastWord(string s)
    {
        int sum = 0;

        s = s.TrimEnd();

        for (int i = 0; i < s.Length; i++)
        {
            if (s[i] == ' ')
            {
                sum = 0;
            }
            else
            {
                sum++;
            }
        }
        return sum;

    }



    static string IntToRomanClean(int num)
    {

        string[] thousandsLookup = new string[] { "", "M", "MM", "MMM" };
        string[] hundredsLookup = new string[] { "", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM" };
        string[] tensLookup = new string[] { "", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC" };
        string[] onesLookup = new string[] { "", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX" };

        int thousands = num / 1000;
        int hundreds = num % 1000 / 100;
        int tens = num % 100 / 10;
        int ones = num % 10;

        return thousandsLookup[thousands] + hundredsLookup[hundreds] + tensLookup[tens] + onesLookup[ones];

    }


    static string IntToRoman(int num)
    {

        string romanValue = "";

        // M  1000's
        if (num >= 1000)
        {

            // input is less than 4000 
            int firstDigit = num / 1000;


            while (firstDigit > 0)
            {
                romanValue += "M";
                firstDigit--;
            }

            num = num % 1000;
        }

        // D or C  100's
        // 900  CM  (Sub)
        // 800  DCCC
        // 700  DCC
        // 600  DC
        // 500  D
        // 400  CD   (Sub)
        // 300  CCC
        // 200  CC
        // 100  C
        if (num >= 100)
        {

            int firstDigit = num / 100;

            // CD CM
            if (firstDigit == 4)
            {
                romanValue += "CD";
                firstDigit -= 4;


            }
            else if (firstDigit == 9)
            {
                romanValue += "CM";
                firstDigit -= 9;


            }
            else
            {

                if (firstDigit >= 5)
                {
                    // append C's to D
                    romanValue += "D";

                    firstDigit -= 5;
                }

                while (firstDigit > 0)
                {
                    romanValue += "C";
                    firstDigit--;
                }


            }

            num = num % 100;
        }

        // X or L  100's
        // 90  XC  (Sub)
        // 80  LXXX
        // 70  LXX
        // 60  LX
        // 50  L
        // 40  XL   (Sub)
        // 30  XXX
        // 20  XX
        // 10  X
        if (num >= 10)
        {

            int firstDigit = num / 10;

            if (firstDigit == 4)
            {
                romanValue += "XL";
                firstDigit -= 4;

            }
            else if (firstDigit == 9)
            {
                romanValue += "XC";
                firstDigit -= 9;

            }
            else
            {

                if (firstDigit >= 5)
                {
                    // append X's to L
                    romanValue += "L";

                    firstDigit -= 5;
                }

                while (firstDigit > 0)
                {
                    romanValue += "X";
                    firstDigit--;
                }


            }

            num = num % 10;

        }

        // i or V
        // IX - Subtractive
        // VIII
        // VII
        // VI
        // V
        // IV  - Subtractive
        // III
        // II
        // I
        if (num >= 1)
        {

            int firstDigit = num % 10;

            if (firstDigit == 4)
            {
                romanValue += "IV";
                firstDigit -= 4;
            }
            else if (firstDigit == 9)
            {
                romanValue += "IX";
                firstDigit -= 9;
            }
            else
            {

                if (firstDigit >= 5)
                {
                    // append I's to V
                    romanValue += "V";
                    firstDigit -= 5;
                }

                while (firstDigit > 0)
                {
                    romanValue += "I";
                    firstDigit--;
                }

            }
        }
        return romanValue;
    }



    static int RomanToIntOptimized(string s)
    {
        Dictionary<char, int> dict = new Dictionary<char, int> {
                { 'I', 1 },
                { 'V', 5 },
                { 'X', 10 },
                { 'L', 50 },
                { 'C', 100 },
                { 'D', 500 },
                { 'M', 1000 }
            };

        int total = 0;
        int prevValue = 0;

        for (int i = s.Length - 1; i >= 0; i--)
        {
            int currentValue = dict[s[i]];

            if (currentValue < prevValue)
            {
                total -= currentValue;
            }
            else
            {
                total += currentValue;
            }

            prevValue = currentValue;
        }

        return total;
    }


    static int RomanToInt(string s)
    {

        Dictionary<string, int> dict = new Dictionary<string, int>();

        dict.Add("I", 1);
        dict.Add("V", 5);
        dict.Add("X", 10);
        dict.Add("L", 50);
        dict.Add("C", 100);
        dict.Add("D", 500);
        dict.Add("M", 1000);

        dict.Add("IV", 4);
        dict.Add("IX", 9);
        dict.Add("XL", 40);
        dict.Add("XC", 90);
        dict.Add("CD", 400);
        dict.Add("CM", 900);

        int i = 0;
        int runningSum = 0;


        while (i < s.Length)
        {

            // first check next pointer if it's a special case
            if (i + 1 < s.Length && dict.ContainsKey(s.Substring(i, 2)))
            {
                runningSum += dict[s.Substring(i, 2)];
                i += 2;
            }
            else
            {
                runningSum += dict[s[i].ToString()];
                i++;
            }

        }

        return runningSum;

    }


    static int TrapOnO1(int[] height)
    {

        // 4th approach
        //
        // O(n) time and O(1) space
        // You always move from the side with the smaller wall
        // That side is the bottleneck, and the other side is at least as tall

        // A left pointer and a right pointer moving toward each other
        // leftMax: the tallest bar you've seen from the left side
        // rightMax: the tallest bar you've seen from the right side
        int left = 0, right = height.Length - 1;
        int leftMax = 0, rightMax = 0;
        int waterSum = 0;

        while (left < right)
        {
            leftMax = Math.Max(leftMax, height[left]);
            rightMax = Math.Max(rightMax, height[right]);

            if (leftMax < rightMax)
            {
                waterSum += leftMax - height[left];
                left++;
            }
            else
            {
                waterSum += rightMax - height[right];
                right--;
            }
        }

        return waterSum;
    }

    static int TrapOnOn(int[] height)
    {

        // water[i] = min(maxLeft, maxRight) - height[i]

        // 3rd approach is O(n) time and O(n) space complexity
        // calculate the max height of the current cell going left
        // and current cell going right
        int[] maxToLeft = new int[height.Length];
        int[] maxToRight = new int[height.Length];

        maxToLeft[0] = height[0];
        for (int i = 1; i < height.Length; i++)
        {
            maxToLeft[i] = Math.Max(maxToLeft[i - 1], height[i]);
        }

        maxToRight[height.Length - 1] = height[height.Length - 1];
        for (int i = height.Length - 2; i >= 0; i--)
        {
            maxToRight[i] = Math.Max(maxToRight[i + 1], height[i]);
        }

        int waterSum = 0;
        // sum up the water level
        for (int i = 0; i < height.Length; i++)
        {
            waterSum += Math.Min(maxToLeft[i], maxToRight[i]) - height[i];
        }


        return waterSum;

    }

    static int TrapOnnO1(int[] height)
    {

        // 2nd Approach is O(n^2) time complexity
        // For each i, mark the "start" of a wall
        // then calculate each "level" which is the max value of the array
        // then count the amount of water inbetween the bounds 
        int trappedWater = 0;
        int maxValue = 0; // TODO: doesn't matter not efficient anyway

        for (int level = 1; level <= maxValue; level++)
        {
            bool started = false;
            int tempWater = 0;
            for (int j = 0; j < height.Length; j++)
            {
                if (height[j] >= level)
                {
                    if (started)
                    {
                        trappedWater += tempWater;
                    }
                    else
                    {
                        started = true;
                    }
                    tempWater = 0;
                }
                else if (started)
                {
                    tempWater++;
                }
            }
        }

        return trappedWater;

    }

    //                            (2,11)
    //        [0,0,0,0,0,0,0,3,0,0,0,0]
    //        [0,0,0,2,0,0,0,3,2,0,2,0]
    //        [0,1,0,2,1,0,1,3,2,1,2,1]
    //       0,0
    //
    //      First idea is to build a 2d array and fill up the array as per example and count between walls
    //      row by row
    //      O(n^2) space and O(n^2) time .. not optimal at all!
    //       static int TrapOnnOnn(int[] height) { ... }



    static int Candy(int[] ratings)
    {
        int n = ratings.Length;
        int[] candies = new int[n];

        // Start with 1 candy for everyone
        for (int i = 0; i < n; i++)
        {
            candies[i] = 1;
        }

        // Go left to right upping the candies
        for (int i = 1; i < n; i++)
        {
            if (ratings[i] > ratings[i - 1])
            {
                candies[i] = candies[i - 1] + 1;
            }
        }

        // go right to left
        for (int i = n - 2; i >= 0; i--)
        {
            if (ratings[i] > ratings[i + 1])
            {
                // You may get a lower rating if you naively do candies[i + 1] + 1, so take the max
                candies[i] = Math.Max(candies[i], candies[i + 1] + 1);
            }
        }
        return candies.Sum();
    }



    // Greedy Approach, keep a tally of the distance we can go
    // If there is guaranteed to be a solution
    // we can discard the index we've chosen and move onto the next one
    static int CanCompleteCircuit(int[] gas, int[] cost)
    {
        int start = 0;
        int tank = 0;
        int total = 0;

        for (int i = 0; i < gas.Length; i++)
        {
            total += gas[i] - cost[i];
        }

        if (total < 0)
        {
            return -1;
        }


        for (int i = 0; i < gas.Length; i++)
        {
            tank += gas[i] - cost[i];

            // This "sub path" fails since we run out
            // If this sub path fails, anything within this subpath will fail as well
            //
            // If you try to start anywhere between start + 1 and i, you’ll have 
            // less gas accumulated than you had starting from start, so you’ll run out of gas even sooner.
            //
            // The only way you can make it beyond "i" is to have more gas along the way, which
            // you did not get anyway otherwise you would have made it.
            if (tank < 0)
            {
                start = i + 1;
                tank = 0;
            }
        }

        return start;
    }


    // It's the same solution as a linear path 
    // with the exception of the last node isn't counted as a cost
    //
    //  int CanReachEnd(int[] gas, int[] cost) {
    //     int start = 0;
    //     int tank = 0;

    //     for (int i = 0; i < gas.Length; i++) {
    //         tank += gas[i] - cost[i];

    //         if (tank < 0) {
    //             // You can't reach i from the current start,
    //             // so skip everything up to i and start fresh from i + 1
    //             start = i + 1;
    //             tank = 0;
    //         }
    //     }

    //     // If start is within bounds, return it
    //     return start < gas.Length ? start : -1;
    // }

    static int[] ProductExceptSelf(int[] nums)
    {
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

        int[] left = new int[nums.Length];
        int[] right = new int[nums.Length];

        left[0] = 1;
        right[nums.Length - 1] = 1;

        for (int i = 1; i < nums.Length; i++)
        {
            //typically
            //nums[i] = nums[i] * nums[i - 1];
            left[i] = left[i - 1] * nums[i - 1];
        }

        for (int i = nums.Length - 2; i >= 0; i--)
        {
            right[i] = right[i + 1] * nums[i + 1];
            // typically an adjustment after to include 
            // a for-loop that includes i
            // right[i] *= right[i];
        }


        for (int i = 0; i < nums.Length; i++)
        {
            output[i] = left[i] * right[i];
        }


        return output;
    }


    // Dictionary<int, int> dictionary;
    // List<int> list;
    // Random random;

    //  RandomizedSet()
    // {
    //     dictionary = new Dictionary<int, int>();
    //     list = new List<int>();
    //     random = new Random();
    // }

    // static bool Insert(int val)
    // {

    //     if (dictionary.ContainsKey(val))
    //     {
    //         return false;
    //     }

    //     list.Add(val);
    //     dictionary.Add(val, list.Count - 1);
    //     return true;
    // }


    //  static bool Remove(int val)
    // {
    //     if (!dictionary.ContainsKey(val))
    //     {
    //         return false;
    //     }

    //     var index = dictionary[val];
    //     var endValue = list[list.Count - 1];

    //     list[index] = endValue; //swap it out

    //     dictionary[endValue] = index;

    //     // remove the value in dictionary
    //     dictionary.Remove(val);

    //     // remove the value in the list
    //     list.RemoveAt(list.Count - 1);

    //     return true;
    // }

    //  static int GetRandom()
    // {
    //     int randomVal = random.Next(0, list.Count);
    //     return list[randomVal];
    // }









    // First we order the values of f from the largest to the lowest value. 
    // Then, we look for the last position in which f 
    // is greater than or equal to the position (we call h this position)
    // Sorting kills the time complexity
    // However! you can use storage to gain O(n) time at the cost of O(n) size
    static int HIndex(int[] citations)
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


    static int JumpII(int[] nums)
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

    static bool CanJump(int[] nums)
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


    static int MaxProfitSpaceOptimized(int[] prices)
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

    static int MaxProfitDP(int[] prices)
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
    static int MaxProfitLinear(int[] prices)
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
    static int MaxProfitMedium(int[] prices)
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
    static int MaxProfit2(int[] prices)
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


    static int MaxProfit(int[] prices)
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
    static void Rotate2(int[] nums, int k)
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
    static void Rotate(int[] nums, int k)
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
    static int BoyerMooreMajority(int[] nums)
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
    static int MajorityElement(int[] nums)
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
    static int RemoveDuplicates2(int[] nums)
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


    static int RemoveDuplicates(int[] nums)
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

    static int RemoveElement(int[] nums, int val)
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




    static void Merge(int[] nums1, int m, int[] nums2, int n)
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