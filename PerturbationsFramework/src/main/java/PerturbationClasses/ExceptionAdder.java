import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.NodeList;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.VariableDeclarator;
import com.github.javaparser.ast.expr.BooleanLiteralExpr;
import com.github.javaparser.ast.expr.NameExpr;
import com.github.javaparser.ast.expr.ObjectCreationExpr;
import com.github.javaparser.ast.expr.StringLiteralExpr;
import com.github.javaparser.ast.expr.VariableDeclarationExpr;
import com.github.javaparser.ast.stmt.BlockStmt;
import com.github.javaparser.ast.stmt.ExpressionStmt;
import com.github.javaparser.ast.stmt.IfStmt;
import com.github.javaparser.ast.stmt.Statement;
import com.github.javaparser.ast.stmt.ThrowStmt;
import com.github.javaparser.ast.type.ClassOrInterfaceType;
import com.github.javaparser.ast.type.PrimitiveType;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

public class ExceptionAdder extends VoidVisitorAdapter<Void> {
    private boolean modificationApplied = false;  // Flag to track if changes were made

    @Override
    public void visit(MethodDeclaration n, Void arg) {
        super.visit(n, arg);
        insertDeadCode(n);
    }

    private void insertDeadCode(MethodDeclaration method) {
        BlockStmt methodBody = method.getBody().orElse(new BlockStmt());

        // Create a boolean variable declaration and initialize it to false
        VariableDeclarator varDeclarator = new VariableDeclarator(
                PrimitiveType.booleanType(), "var", new BooleanLiteralExpr(false));
        VariableDeclarationExpr varDeclExpr = new VariableDeclarationExpr(varDeclarator);
        ExpressionStmt varDeclStmt = new ExpressionStmt(varDeclExpr);  // Wrap in an ExpressionStmt

        // Create the if statement with the dead code
        IfStmt ifStmt = new IfStmt(
                new NameExpr("var"),
                new ThrowStmt(new ObjectCreationExpr(
                        null,
                        new ClassOrInterfaceType(null, "RuntimeException"),
                        NodeList.nodeList(new StringLiteralExpr("Unexpected condition encountered"))
                )),
                null
        );

        // Insert the dead code at the beginning of the method body
        NodeList<Statement> statements = new NodeList<>();
        statements.add(varDeclStmt);
        statements.add(ifStmt);
        statements.addAll(methodBody.getStatements());
        methodBody.setStatements(statements);
        method.setBody(methodBody);

        modificationApplied = true;  // Mark as modified
    }

    public boolean isModified() {
        return modificationApplied;
    }
}
