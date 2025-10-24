import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class CodeFormatter {
    public static String formatCode(String inputCode) {
        // Remove the class declaration and ending braces
        String strippedCode = inputCode.replaceAll("public class DummyClass[^{]+\\{", "")
                .replaceAll("\\}\\s*$", "");

        // Remove nested empty braces
        strippedCode = strippedCode.replaceAll("\\{\\s*\\{\\s*\\}\\s*\\}", "{ }");

        // Remove indentations and extra white spaces
        strippedCode = strippedCode.replaceAll("\\s+", " ");

        return strippedCode.trim();
    }


    public static String insertStartEndTags(String sourceCode) {
        String regex = "(returnValue\\s*=\\s*)(.*?);";
        Pattern pattern = Pattern.compile(regex);
        Matcher matcher = pattern.matcher(sourceCode);
        StringBuffer result = new StringBuffer();

        while (matcher.find()) {
            String segmentBefore = matcher.group(1);
            String segmentToModify = matcher.group(2);
            segmentToModify = segmentToModify.replaceAll("\\/\\*<START>\\*\\/", "").replaceAll("\\/\\*<END>\\*\\/", "");
            String modifiedAssignment = segmentBefore + " /*<START>*/ " + segmentToModify + " /*<END>*/;";
            matcher.appendReplacement(result, Matcher.quoteReplacement(modifiedAssignment));
        }
        matcher.appendTail(result);
        return result.toString();
    }



    private static String removeDuplicatedTags(String sourceCode, String tag) {
        int firstIndex = sourceCode.indexOf(tag);
        int secondIndex = sourceCode.indexOf(tag, firstIndex + tag.length());

        if (firstIndex != -1 && secondIndex != -1) {
            return sourceCode.substring(0, firstIndex) + sourceCode.substring(firstIndex + tag.length());
        }

        return sourceCode;
    }


    public static String convertTagsWithOrder(String code) {
        String startTag = "/*<START>*/";
        String endTag = "/*<END>*/";

        code = insertStartEndTags(code);
        code = removeDuplicatedTags(code, startTag);
        code = removeDuplicatedTags(code, endTag);

        if (!code.contains(startTag)) {
            code = startTag + code;
        }
        if (!code.contains(endTag)) {
            code = code + endTag;
        }

        int startIndex = code.indexOf(startTag);
        int endIndex = code.indexOf(endTag);

        // Check if both tags are present and START comes after END
        if (startIndex != -1 && endIndex != -1 && startIndex > endIndex) {

            // Extract the parts of the code before, between, and after the tags
            String beforeEnd = code.substring(0, endIndex);
            String betweenTags = code.substring(endIndex + endTag.length(), startIndex);
            String afterStart = code.substring(startIndex + startTag.length());

            // Reconstruct the string with switched tags
            code = beforeEnd + startTag + betweenTags + endTag + afterStart;
        }

        // Replace the tags in the potentially modified string
        return code.replace(startTag, "<START>").replace(endTag, "<END>");
    }

}
