import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.*;
import com.github.javaparser.ast.stmt.*;
import com.github.javaparser.ast.type.Type;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.github.javaparser.ast.NodeList;
import com.github.javaparser.ast.body.VariableDeclarator;



import java.util.ArrayList;
import java.util.List;

public class DataFlowChangeT extends VoidVisitorAdapter<Void> {
    private boolean modified = false;

    @Override
    public void visit(ClassOrInterfaceDeclaration n, Void arg) {
        super.visit(n, arg);
        n.getMethods().forEach(this::processMethod);
    }

    private void processMethod(MethodDeclaration method) {
        if (method.getBody().isPresent()) {
            NodeList<Statement> statements = method.getBody().get().getStatements();
            // Collect all return statements first to avoid ConcurrentModificationException
            List<ReturnStmt> returnStmts = new ArrayList<>();
            for (Statement stmt : statements) {
                if (stmt instanceof ReturnStmt) {
                    returnStmts.add((ReturnStmt) stmt);
                }
            }
            // Now process each return statement separately
            for (ReturnStmt returnStmt : returnStmts) {
                processReturnStatement(method, returnStmt);
            }
        }
    }

    private void processReturnStatement(MethodDeclaration method, ReturnStmt returnStmt) {
        Expression returnExpr = returnStmt.getExpression().orElse(new NullLiteralExpr());
        String varName = "returnValue";
        Type returnType = method.getType();

        BlockStmt methodBlock = method.getBody().get();

        // Create the new variable declaration and assignment
        VariableDeclarationExpr varDecl = new VariableDeclarationExpr(new VariableDeclarator(returnType, varName));
        AssignExpr assignExpr = new AssignExpr(varDecl, returnExpr, AssignExpr.Operator.ASSIGN);

        // Add new assignment before the return statement
        methodBlock.getStatements().add(methodBlock.getStatements().indexOf(returnStmt), new ExpressionStmt(assignExpr));

        // Replace the return statement to return the new variable
        returnStmt.setExpression(new NameExpr(varName));

        modified = true;
    }

    public boolean isModified() {
        return modified;
    }
}
