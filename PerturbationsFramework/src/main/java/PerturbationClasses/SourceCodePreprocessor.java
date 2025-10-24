import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class SourceCodePreprocessor {

    public static String processSourceCode(String sourceCode, String taggedSource) {
        // Extract positions to insert tags
        int startPosition = getWordPosition(taggedSource, "<START>") + 4; // Adjusted position for <START>
        int endPosition = getWordPosition(taggedSource, "<END>") + 4;    // Adjusted position for <END>

        System.out.println("Insert <START> tag at word position: " + startPosition);
        System.out.println("Insert <END> tag at word position: " + endPosition);

        // Insert the tags in the source code
        String processedSourceCodeI = insertTagAtWordPosition(sourceCode, startPosition, " /*<START>*/ " );
        String processedSourceCodeII = insertTagAtWordPosition(processedSourceCodeI, endPosition, " /*<END>*/ " );


        return processedSourceCodeII;
    }

    public static int getWordPosition(String taggedSource, String tag) {
        String[] tokens = taggedSource.split("\\s+"); // Split by whitespace to get tokens
        for (int i = 0; i < tokens.length; i++) {
            if (tokens[i].equals(tag)) {
                return i;
            }
        }
        // Return 0 if <START> is not found, tokens.length if <END> is not found
        if (tag.equals("<START>")) {
            return 0;
        } else if (tag.equals("<END>")) {
            return tokens.length;
        }
        throw new IllegalArgumentException("Unsupported tag: " + tag);
    }


    private static String insertTagAtWordPosition(String code, int startWordPosition, String startTag) {
        StringBuilder result = new StringBuilder();
        int wordCount = 0; // Tracks the position of words
        boolean startTagInserted = false;

        for (int i = 0; i < code.length(); i++) {
            char c = code.charAt(i);

            // Identify word boundaries
            if (Character.isWhitespace(c)) {
                if (i > 0 && !Character.isWhitespace(code.charAt(i - 1))) {
                    wordCount++; // Increment word count after a word ends
                }
            }

            // Insert <START> tag at the correct position
            if (!startTagInserted && wordCount == startWordPosition) {
                result.append(startTag);
                startTagInserted = true;
            }

            // Append the current character to the result
            result.append(c);
        }

        return result.toString();
    }
}
