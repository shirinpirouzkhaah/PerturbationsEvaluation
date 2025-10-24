import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class FixTags {

    public static class MatchResult {
        public boolean found;
        public String modifiedTarget;

        public MatchResult(boolean found, String modifiedTarget) {
            this.found = found;
            this.modifiedTarget = modifiedTarget;
        }
    }

    public static String normalizeString(String input) {
        // Normalize spaces, remove all new lines, and remove space following commas
        input = input.replaceAll("\\s+", " ").trim(); // Replace all whitespace with a single space
        input = input.replaceAll(",\\s+", ","); // Remove space after commas
        return input;
    }


    public static MatchResult matchSubstring(String target, String taggedSubstring, String randomName, String varName) {
        target = removeCommentTags(target);
        // Replace specific variable names if necessary
        if (varName != null && randomName != null) {
            taggedSubstring = taggedSubstring.replaceAll("\\b" + Pattern.quote(varName) + "\\b", Matcher.quoteReplacement(randomName));
        }

        target = normalizeString(target);
        taggedSubstring = normalizeString(taggedSubstring);
        String regexSafeTaggedSubstring = Pattern.quote(taggedSubstring);
        Pattern pattern = Pattern.compile(regexSafeTaggedSubstring);
        Matcher matcher = pattern.matcher(target);

        if (matcher.find()) {
            String modifiedTarget = insertTags(target, matcher);
            return new MatchResult(true, modifiedTarget);
        } else {
            return new MatchResult(false, target);
        }
    }

    public static String insertTags(String target, Matcher matcher) {
        StringBuilder builder = new StringBuilder(target);
        // Correctly adjust for subsequent tags after modifying the StringBuilder
        builder.insert(matcher.end(), "<END>");
        builder.insert(matcher.start(), "<START>");
        return builder.toString();
    }


    public static String removeCommentTags(String input) {
        // Remove the specific comment tags and optionally any new lines immediately following them
        return input.replaceAll("\\/\\*<START>\\*\\/(\\r?\\n)?", "")
                .replaceAll("\\/\\*<END>\\*\\/(\\r?\\n)?", "");
    }



    public static String extractSubstring(String input) {
        int startTag = input.indexOf("<START>") + "<START>".length();
        int endTag = input.indexOf("<END>");
        if (startTag < 0 || endTag < 0 || endTag < startTag) {
            startTag = 0; // Fallback to the entire string if tags are incorrectly placed
            endTag = input.length() - 1;
        }
        return input.substring(startTag, endTag).trim();
    }
}
