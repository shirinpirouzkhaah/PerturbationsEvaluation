import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.NodeList;
import com.github.javaparser.ast.stmt.CatchClause;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.Parameter;
import com.github.javaparser.ast.expr.StringLiteralExpr;
import com.github.javaparser.ast.stmt.BlockStmt;
import com.github.javaparser.ast.stmt.Statement;
import com.github.javaparser.ast.stmt.ThrowStmt;
import com.github.javaparser.ast.stmt.TryStmt;
import com.github.javaparser.ast.type.ClassOrInterfaceType;
import com.github.javaparser.ast.type.Type;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

import com.github.javaparser.ast.expr.BinaryExpr;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.expr.ObjectCreationExpr;
import com.github.javaparser.ast.stmt.ForStmt;
import com.github.javaparser.ast.stmt.WhileStmt;
import com.github.javaparser.ast.stmt.DoStmt;


public class TryCatchModifier extends VoidVisitorAdapter<Void> {
    private boolean modified = false;

    @Override
    public void visit(ClassOrInterfaceDeclaration n, Void arg) {
        super.visit(n, arg);
        n.getMethods().forEach(method -> {
            method.getBody().ifPresent(body -> {
                boolean hasTryCatch = body.findAll(TryStmt.class).size() > 0;
                if (!body.getStatements().isEmpty() && !hasTryCatch) {
                    addTryCatch(method);
                    modified = true;
                }
            });
        });
    }


    private void addTryCatch(MethodDeclaration method) {
        BlockStmt tryBlock = new BlockStmt(new NodeList<>(method.getBody().get().getStatements()));

        TryStmt tryStmt = new TryStmt();
        tryStmt.setTryBlock(tryBlock);

        BlockStmt catchBlock = new BlockStmt();
        catchBlock.addStatement(new ThrowStmt(new com.github.javaparser.ast.expr.NameExpr("e")));

        ClassOrInterfaceType exceptionType = new ClassOrInterfaceType(null, "Exception");
        CatchClause catchClause = new CatchClause(
                new Parameter(exceptionType, "e"), catchBlock
        );

        tryStmt.getCatchClauses().add(catchClause);
        method.setBody(new BlockStmt().addStatement(tryStmt));
    }

    public boolean isModified() {
        return modified;
    }
}
