import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.AssignExpr;
import com.github.javaparser.ast.expr.Expression;
import com.github.javaparser.ast.expr.NameExpr;
import com.github.javaparser.ast.expr.VariableDeclarationExpr;
import com.github.javaparser.ast.stmt.BlockStmt;
import com.github.javaparser.ast.stmt.ExpressionStmt;
import com.github.javaparser.ast.stmt.ReturnStmt;
import com.github.javaparser.ast.stmt.Statement;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.github.javaparser.ast.expr.MethodCallExpr;


import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

public class IndependentLineSwap extends VoidVisitorAdapter<Void> {
    private boolean modified = false;

    @Override
    public void visit(ClassOrInterfaceDeclaration n, Void arg) {
        super.visit(n, arg);
        n.getMethods().forEach(this::swapIndependentStatements);
    }

    private void swapIndependentStatements(MethodDeclaration method) {
        if (!method.getBody().isPresent()) return;

        BlockStmt body = method.getBody().get();
        List<Statement> statements = body.getStatements();
        Set<Expression> returnExpressions = method.findAll(ReturnStmt.class).stream()
                .map(ReturnStmt::getExpression)
                .filter(Optional::isPresent)
                .map(Optional::get)
                .collect(Collectors.toSet());

        for (int i = 0; i < statements.size() - 1; i++) {
            Statement first = statements.get(i);
            Statement second = statements.get(i + 1);

            if (isSwapSafe(first, second)) {
                // Swap statements
                statements.set(i, second);
                statements.set(i + 1, first);
                modified = true;
                break; // Swap only the first pair found
            }
        }
    }

    private boolean isSwapSafe(Statement first, Statement second) {
        return (isIndependentVariableDeclaration(first, second) ||
                isIndependentAssignment(first, second)) &&
                !containsMethodCall(second);
    }

    private boolean containsMethodCall(Statement stmt) {
        if (stmt instanceof ExpressionStmt) {
            Expression expr = ((ExpressionStmt) stmt).getExpression();
            return expr.findAll(MethodCallExpr.class).size() > 0;
        }
        return false;
    }

    private boolean isIndependentVariableDeclaration(Statement first, Statement second) {
        if (!(first instanceof ExpressionStmt && ((ExpressionStmt) first).getExpression() instanceof VariableDeclarationExpr)) {
            return false;
        }
        if (!(second instanceof ExpressionStmt && ((ExpressionStmt) second).getExpression() instanceof VariableDeclarationExpr)) {
            return false;
        }

        VariableDeclarationExpr firstDecl = (VariableDeclarationExpr) ((ExpressionStmt) first).getExpression();
        VariableDeclarationExpr secondDecl = (VariableDeclarationExpr) ((ExpressionStmt) second).getExpression();

        Set<String> firstVars = firstDecl.getVariables().stream()
                .map(v -> v.getNameAsString())
                .collect(Collectors.toSet());

        Set<String> secondVars = secondDecl.getVariables().stream()
                .map(v -> v.getNameAsString())
                .collect(Collectors.toSet());

        // Ensure none of the variables in secondDecl's initializers depend on any variables declared in firstDecl
        return secondDecl.getVariables().stream().noneMatch(v ->
                v.getInitializer().isPresent() &&
                        v.getInitializer().get().findAll(NameExpr.class).stream()
                                .map(NameExpr::getNameAsString)
                                .anyMatch(firstVars::contains));
    }

    private boolean isIndependentAssignment(Statement first, Statement second) {
        if (!(first instanceof ExpressionStmt && ((ExpressionStmt) first).getExpression() instanceof AssignExpr)) {
            return false;
        }
        if (!(second instanceof ExpressionStmt && ((ExpressionStmt) second).getExpression() instanceof AssignExpr)) {
            return false;
        }

        AssignExpr firstAssign = (AssignExpr) ((ExpressionStmt) first).getExpression();
        AssignExpr secondAssign = (AssignExpr) ((ExpressionStmt) second).getExpression();

        Set<String> firstVars = getVariablesUsed(firstAssign.getValue());
        Set<String> secondVars = getVariablesUsed(secondAssign.getValue());

        return !firstVars.contains(secondAssign.getTarget().toString()) &&
                !secondVars.contains(firstAssign.getTarget().toString());
    }

    private boolean isReturned(Statement stmt, Set<Expression> returnExpressions) {
        Set<String> assignedVariables = getVariablesAssigned(stmt);
        return returnExpressions.stream()
                .flatMap(expr -> expr.findAll(NameExpr.class).stream())
                .map(NameExpr::getNameAsString)
                .anyMatch(assignedVariables::contains);
    }

    private Set<String> getVariablesAssigned(Statement stmt) {
        Set<String> assignedVariables = new HashSet<>();
        if (stmt instanceof ExpressionStmt) {
            Expression expr = ((ExpressionStmt) stmt).getExpression();
            if (expr instanceof AssignExpr) {
                assignedVariables.add(((AssignExpr) expr).getTarget().toString());
            } else if (expr instanceof VariableDeclarationExpr) {
                assignedVariables.addAll(((VariableDeclarationExpr) expr).getVariables().stream()
                        .map(v -> v.getNameAsString())
                        .collect(Collectors.toSet()));
            }
        }
        return assignedVariables;
    }

    private Set<String> getVariablesUsed(Expression expr) {
        return expr.findAll(NameExpr.class).stream()
                .map(NameExpr::getNameAsString)
                .collect(Collectors.toSet());
    }

    public boolean isModified() {
        return modified;
    }
}
