import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.FieldDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.Parameter;
import com.github.javaparser.ast.body.VariableDeclarator;
import com.github.javaparser.ast.expr.StringLiteralExpr;
import com.github.javaparser.ast.comments.Comment;
import com.github.javaparser.ast.expr.SimpleName;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

import java.util.Collections;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.util.Random;
import java.util.Set;
import java.util.HashSet;

public class VariableShuffler extends VoidVisitorAdapter<Void> {
    private Map<String, String> nameMapping = new HashMap<>();
    private boolean modified = false;

    // Modify the method to return the modified comment
    public String shuffleVariables(CompilationUnit sourceCu, CompilationUnit targetCu, String reviewComment) {
        // Generate name mapping from source CU
        if (createNameMapping(sourceCu)) {
            // Apply mapping to source CU
            applyMappingToUnit(sourceCu);

            // Apply the same mapping to target CU
            applyMappingToUnit(targetCu);
        }

        // Replace names in the review comment and return it
        return replaceInComment(reviewComment);
    }

    private boolean createNameMapping(CompilationUnit cu) {
        Set<String> originalNames = new HashSet<>();
        cu.findAll(VariableDeclarator.class).forEach(varDecl -> originalNames.add(varDecl.getNameAsString()));
        cu.findAll(Parameter.class).forEach(param -> originalNames.add(param.getNameAsString()));
        cu.findAll(FieldDeclaration.class).forEach(field -> field.getVariables().forEach(var -> originalNames.add(var.getNameAsString())));

        // Proceed only if there are more than one unique names
        if (originalNames.size() > 1) {
            List<String> originalList = new ArrayList<>(originalNames);
            List<String> shuffledNames = new ArrayList<>(originalList);
            Collections.shuffle(shuffledNames, new Random());  // Shuffle the names
            for (int i = 0; i < originalList.size(); i++) {
                nameMapping.put(originalList.get(i), shuffledNames.get(i));
            }
            return true;
        }
        return false;
    }

    private void applyMappingToUnit(CompilationUnit cu) {
        cu.findAll(SimpleName.class).forEach(name -> {
            String newName = nameMapping.get(name.getIdentifier());
            if (newName != null) {
                name.setIdentifier(newName);
                modified = true;
            }
        });

        cu.findAll(StringLiteralExpr.class).forEach(literal -> replaceInStringLiteral(literal));
        cu.findAll(Comment.class).forEach(comment -> comment.setContent(replaceInComment(comment.getContent())));
    }

    private void replaceInStringLiteral(StringLiteralExpr literal) {
        String modifiedLiteral = literal.getValue();
        for (Map.Entry<String, String> entry : nameMapping.entrySet()) {
            modifiedLiteral = modifiedLiteral.replaceAll("\\b" + entry.getKey() + "\\b", entry.getValue());
        }
        literal.setValue(modifiedLiteral);
    }

    private String replaceInComment(String comment) {
        String modifiedComment = comment;
        for (Map.Entry<String, String> entry : nameMapping.entrySet()) {
            modifiedComment = modifiedComment.replaceAll("\\b" + entry.getKey() + "\\b", entry.getValue());
        }
        return modifiedComment;
    }

    public boolean isModified() {
        return modified;
    }

    public Map<String, String> getNameMapping() {
        return nameMapping;
    }
}
