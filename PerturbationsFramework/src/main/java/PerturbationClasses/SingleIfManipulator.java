import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.expr.Expression;
import com.github.javaparser.ast.expr.UnaryExpr;
import com.github.javaparser.ast.expr.EnclosedExpr;
import com.github.javaparser.ast.stmt.BlockStmt;
import com.github.javaparser.ast.stmt.IfStmt;
import com.github.javaparser.ast.stmt.Statement;
import com.github.javaparser.ast.visitor.ModifierVisitor;

public class SingleIfManipulator extends ModifierVisitor<Void> {
    private boolean Singlemodified = false;

    @Override
    public IfStmt visit(IfStmt n, Void arg) {
        super.visit(n, arg);
        if (!n.getElseStmt().isPresent() && !new IfManipulator().containsNestedIfElse(n.getThenStmt())) {
            Singlemodified = true;
            // Wrap the original condition in parentheses before negating
            Expression enclosedCondition = new EnclosedExpr(n.getCondition());
            Expression negatedCondition = new UnaryExpr(enclosedCondition, UnaryExpr.Operator.LOGICAL_COMPLEMENT);
            n.setCondition(negatedCondition);

            Statement originalIfBody = n.getThenStmt();

            // Ensure we're working with a BlockStmt directly
            if (!(originalIfBody instanceof BlockStmt)) {
                originalIfBody = new BlockStmt().addStatement(originalIfBody.clone());
            }

            BlockStmt dummyBody = new BlockStmt();
            // Add an empty BlockStmt as a do-nothing action
            dummyBody.addStatement(new BlockStmt()); // This is effectively a do-nothing statement in JavaParser

            // Set the dummy body as the new 'then' block
            n.setThenStmt(dummyBody);

            // Use the original 'if' body directly as the 'else' block
            n.setElseStmt(originalIfBody);
        }
        return n;
    }

    public boolean SingleisModified() {
        return Singlemodified;
    }
}
