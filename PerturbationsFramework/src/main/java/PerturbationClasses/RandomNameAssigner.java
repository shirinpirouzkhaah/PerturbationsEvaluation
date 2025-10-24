import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.FieldDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.Parameter;
import com.github.javaparser.ast.body.VariableDeclarator;
import com.github.javaparser.ast.expr.SimpleName;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.util.Random;

public class RandomNameAssigner extends VoidVisitorAdapter<Void> {
    private Map<String, String> nameMapping = new HashMap<>();
    private boolean modified = false;
    private Random random = new Random();

    public String assignRandomNames(CompilationUnit sourceCu, CompilationUnit targetCu, String reviewComment) {
        // Generate name mapping from source CU
        createNameMapping(sourceCu);

        // Apply mapping to source CU
        applyMappingToUnit(sourceCu);

        // Apply the same mapping to target CU
        applyMappingToUnit(targetCu);

        // Replace names in the review comment
        return replaceInComment(reviewComment);
    }

    private void createNameMapping(CompilationUnit cu) {
        List<String> originalNames = new ArrayList<>();
        cu.findAll(VariableDeclarator.class).forEach(varDecl -> addNameIfAbsent(varDecl.getNameAsString()));
        cu.findAll(Parameter.class).forEach(param -> addNameIfAbsent(param.getNameAsString()));
        cu.findAll(FieldDeclaration.class).forEach(field -> field.getVariables().forEach(var -> addNameIfAbsent(var.getNameAsString())));
    }

    private void addNameIfAbsent(String name) {
        if (!nameMapping.containsKey(name)) {
            nameMapping.put(name, generateRandomWord(5));
        }
    }

    private void applyMappingToUnit(CompilationUnit cu) {
        cu.findAll(SimpleName.class).forEach(name -> {
            String newName = nameMapping.get(name.getIdentifier());
            if (newName != null) {
                name.setIdentifier(newName);
                modified = true;
            }
        });
    }

    private String generateRandomWord(int length) {
        return random.ints('a', 'z' + 1)
                .limit(length)
                .collect(StringBuilder::new, StringBuilder::appendCodePoint, StringBuilder::append)
                .toString();
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

