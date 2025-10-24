import com.github.javaparser.ast.body.VariableDeclarator;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.github.javaparser.ast.expr.VariableDeclarationExpr;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.expr.NameExpr;
import com.github.javaparser.ast.stmt.BlockStmt;
import com.github.javaparser.ast.stmt.ExpressionStmt;
import java.util.concurrent.ThreadLocalRandom;

public class DefUseBreakMutator extends VoidVisitorAdapter<Void> {
    private boolean modified = false;
    private String predefinedName = null;  // Allows setting a predefined random name
    private String lastRandomNameUsed = null;  // Store the last random name generated
    private String OriginalName = null;


    public DefUseBreakMutator() {
    }

    public DefUseBreakMutator(String predefinedName) {
        this.predefinedName = predefinedName;  // Constructor to set predefined name
    }

    @Override
    public void visit(MethodDeclaration n, Void arg) {
        super.visit(n, arg);

        n.findAll(VariableDeclarationExpr.class).stream()
                .findFirst()
                .ifPresent(vde -> {
                    if (!vde.getVariables().isEmpty()) {
                        VariableDeclarator original = vde.getVariables().get(0);
                        if (original.getInitializer().isPresent()) {
                            String originalName = original.getNameAsString();
                            String randomName = (predefinedName != null) ? predefinedName : generateRandomName();
                            lastRandomNameUsed = randomName;
                            OriginalName = originalName;
                            // Rest of the code as previously
                            VariableDeclarator clone = new VariableDeclarator(original.getType(), randomName, original.getInitializer().get());
                            VariableDeclarationExpr newDecl = new VariableDeclarationExpr(clone);
                            ExpressionStmt newStmt = new ExpressionStmt(newDecl);
                            Node parent = vde.findAncestor(BlockStmt.class).orElse(null);
                            if (parent instanceof BlockStmt) {
                                BlockStmt block = (BlockStmt) parent;
                                block.getStatements().stream()
                                        .filter(stmt -> stmt.isExpressionStmt() && stmt.asExpressionStmt().getExpression() == vde)
                                        .findFirst()
                                        .ifPresent(stmt -> {
                                            int index = block.getStatements().indexOf(stmt);
                                            block.addStatement(index + 1, newStmt);
                                            modified = true;
                                        });
                            }
                            n.findAll(NameExpr.class).forEach(nameExpr -> {
                                if (nameExpr.getNameAsString().equals(originalName)) {
                                    nameExpr.setName(randomName);
                                    modified = true;
                                }
                            });
                        }
                    }
                });
    }

    public boolean isModified() {
        return modified;
    }
    public String getLastRandomNameUsed() {
        return lastRandomNameUsed;
    }

    public String OrgName() {
        return OriginalName;
    }


    private String generateRandomName() {
        return ThreadLocalRandom.current().ints('a', 'z' + 1)
                .limit(5)
                .collect(StringBuilder::new, StringBuilder::appendCodePoint, StringBuilder::append)
                .toString();
    }
}
