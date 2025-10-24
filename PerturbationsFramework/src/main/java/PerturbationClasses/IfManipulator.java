import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.NodeList;
import com.github.javaparser.ast.expr.Expression;
import com.github.javaparser.ast.expr.BinaryExpr;
import com.github.javaparser.ast.expr.UnaryExpr;
import com.github.javaparser.ast.stmt.BlockStmt;
import com.github.javaparser.ast.stmt.ExpressionStmt;
import com.github.javaparser.ast.stmt.IfStmt;
import com.github.javaparser.ast.stmt.Statement;
import com.github.javaparser.ast.visitor.ModifierVisitor;
import com.github.javaparser.ast.expr.EnclosedExpr;

public class IfManipulator extends ModifierVisitor<Void> {
    private boolean modified = false;
    @Override
    public IfStmt visit(IfStmt n, Void arg) {
        super.visit(n, arg);
        if (n.getElseStmt().isPresent() || containsNestedIfElse(n.getThenStmt())) {
            modified = true;
            // Handle the then statement
            Statement originalIfBody = n.getThenStmt();
            if (!(originalIfBody instanceof BlockStmt) && n.getElseStmt().isPresent()) {
                originalIfBody = new BlockStmt(new NodeList<>(originalIfBody)); // Wrap only if necessary
            }

            // Handle the else statement
            Statement elseStmt = n.getElseStmt().orElse(new BlockStmt());
            if (!(elseStmt instanceof BlockStmt)) {
                elseStmt = new BlockStmt(new NodeList<>(elseStmt)); // Wrap only if necessary
            }

            // Negate the condition
            Expression negatedCondition = negateExpression(n.getCondition());

            // Swap the then and else parts
            n.setThenStmt(elseStmt);
            n.setElseStmt(originalIfBody);
            n.setCondition(negatedCondition);
        }
        return n;
    }

    public boolean isModified() {
        return modified;
    }

    public boolean containsIfElse(CompilationUnit cu) {
        return cu.findAll(IfStmt.class).stream().anyMatch(ifStmt -> ifStmt.getElseStmt().isPresent() || containsNestedIfElse(ifStmt.getThenStmt()));
    }

    public boolean containsNestedIfElse(Statement stmt) {
        if (stmt instanceof IfStmt) {
            IfStmt ifStmt = (IfStmt) stmt;
            return ifStmt.getElseStmt().isPresent() || containsNestedIfElse(ifStmt.getThenStmt()) ||
                    ifStmt.getElseStmt().map(this::containsNestedIfElse).orElse(false);
        } else if (stmt instanceof BlockStmt) {
            BlockStmt block = (BlockStmt) stmt;
            return block.getStatements().stream().anyMatch(this::containsNestedIfElse);
        }
        return false;
    }

    private Expression negateExpression(Expression expr) {
        Expression enclosedExpression = new EnclosedExpr(expr); // Wrap expression in parentheses before negation

        if (expr instanceof BinaryExpr) {
            BinaryExpr binaryExpr = (BinaryExpr) expr;
            BinaryExpr.Operator operator = binaryExpr.getOperator();
            Expression left = binaryExpr.getLeft();
            Expression right = binaryExpr.getRight();

            switch (operator) {
                case EQUALS:
                    return new UnaryExpr(new BinaryExpr(left, right, BinaryExpr.Operator.NOT_EQUALS), UnaryExpr.Operator.LOGICAL_COMPLEMENT);
                case NOT_EQUALS:
                    return new UnaryExpr(new BinaryExpr(left, right, BinaryExpr.Operator.EQUALS), UnaryExpr.Operator.LOGICAL_COMPLEMENT);
                default:
                    return new UnaryExpr(enclosedExpression, UnaryExpr.Operator.LOGICAL_COMPLEMENT);
            }
        } else if (expr instanceof UnaryExpr) {
            UnaryExpr unaryExpr = (UnaryExpr) expr;
            if (unaryExpr.getOperator() == UnaryExpr.Operator.LOGICAL_COMPLEMENT) {
                return unaryExpr.getExpression(); // Remove extra negation if it's already negated
            } else {
                return new UnaryExpr(enclosedExpression, UnaryExpr.Operator.LOGICAL_COMPLEMENT);
            }
        } else {
            return new UnaryExpr(enclosedExpression, UnaryExpr.Operator.LOGICAL_COMPLEMENT);
        }
    }
}
