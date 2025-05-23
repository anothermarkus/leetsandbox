using System.Text;
using System.Collections;

static class LeetChallenges
{

   public static void SetZeroesNoSpace(int[][] matrix) {
        int rows = matrix.Length;
        int cols = matrix[0].Length;

        bool firstRowZero = false;
        bool firstColZero = false;

        // Check if first row has any zeros
        for (int j = 0; j < cols; j++) {
            if (matrix[0][j] == 0) {
                firstRowZero = true;
                break;
            }
        }

        // Check if first column has any zeros
        for (int i = 0; i < rows; i++) {
            if (matrix[i][0] == 0) {
                firstColZero = true;
                break;
            }
        }

        // Use first row and column as markers
        for (int i = 1; i < rows; i++) {
            for (int j = 1; j < cols; j++) {
                if (matrix[i][j] == 0) {
                    matrix[i][0] = 0;
                    matrix[0][j] = 0;
                }
            }
        }

        // Zero out cells based on markers
        for (int i = 1; i < rows; i++) {
            for (int j = 1; j < cols; j++) {
                if (matrix[i][0] == 0 || matrix[0][j] == 0) {
                    matrix[i][j] = 0;
                }
            }
        }

        // Zero out first row if needed
        if (firstRowZero) {
            for (int j = 0; j < cols; j++) {
                matrix[0][j] = 0;
            }
        }

        // Zero out first column if needed
        if (firstColZero) {
            for (int i = 0; i < rows; i++) {
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

    private static bool IsSubSolutionValid(char[][] board, int startRow, int endRow, int startCol, int endCol) {
        HashSet<char> setOfNine  = new HashSet<char>();

        for (int i = startRow; i < endRow; i++) {
            for (int j = startCol; j < endCol; j++) {
                char candidate = board[i][j];
                if (candidate != '.') {
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


     public static IList<IList<int>> ThreeSumPointer(int[] nums) {
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

     public static IList<IList<int>> ThreeSumHash(int[] nums) {
        var results = new HashSet<string>(); // Use a set to prevent duplicates
        var final = new List<IList<int>>();

        for (int i = 0; i < nums.Length; i++) {
            int fixedNum = nums[i];
            var seen = new HashSet<int>();

            for (int j = i + 1; j < nums.Length; j++) {
                int complement = -fixedNum - nums[j];

                if (seen.Contains(complement)) {
                    var triplet = new List<int> { fixedNum, nums[j], complement };
                    triplet.Sort(); // Ensure triplets are always in same order for deduplication
                    string key = string.Join(",", triplet);
                    if (!results.Contains(key)) {
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
    public static int[] TwoSum(int[] numbers, int target) {
        int left = 0;
        int right = numbers.Length - 1;

        while (left < right) {
            int sum = numbers[left] + numbers[right];
            if (sum == target) {
                return new int[] { left + 1, right + 1 }; 
            }
            if (sum < target) {
                left++;
            } else {
                right--;
            }
        }

        return new int[0]; 
    }


    public static bool IsSubsequence(string s, string t) {
        int schar = 0;
        int tchar = 0;

        while (schar < s.Length && tchar < t.Length) {
            if (s[schar] == t[tchar]) {
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
            {'I', 1},
            {'V', 5},
            {'X', 10},
            {'L', 50},
            {'C', 100},
            {'D', 500},
            {'M', 1000}
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