import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseResult;
import com.github.javaparser.ast.CompilationUnit;


import java.io.FileReader;
import java.io.IOException;

import java.util.regex.Pattern;
import java.util.regex.Matcher;

public class JavaCodeAnalyzer {
    public static void main(String[] args) {
        String filePath = "TufanoDataset.csv";
        String outputPath = "Perturbations.csv";

        // Reading the CSV file and processing each record
        try (FileReader fileReader = new FileReader(filePath);
             CSVParser parser = new CSVParser(fileReader, CSVFormat.DEFAULT.withFirstRecordAsHeader().withTrim())) {

            List<CSVRecord> records = new ArrayList<>(parser.getRecords());
            List<String> headersList = new ArrayList<>(parser.getHeaderMap().keySet());

            headersList.add("DataTypeCFSource");
            headersList.add("DataTypeCFComment");
            headersList.add("DataTypeCFTarget");


            headersList.add("IfElseCFSource");
            headersList.add("IfElseCFComment");
            headersList.add("IfElseCFTarget");

            headersList.add("ExceptionCFSource");
            headersList.add("ExceptionCFComment");
            headersList.add("ExceptionCFTarget");

            headersList.add("DeadCodeCFSource");
            headersList.add("DeadCodeCFComment");
            headersList.add("DeadCodeCFTarget");

            headersList.add("TryNcatchCFSource");
            headersList.add("TryNcatchCFComment");
            headersList.add("TryNcatchCFTarget");

            headersList.add("DataFlowCFSource");
            headersList.add("DataFlowCFComment");
            headersList.add("DataFlowCFTarget");

            headersList.add("EqualAssertCFSource");
            headersList.add("EqualAssertCFComment");
            headersList.add("EqualAssertCFTarget");

            headersList.add("NullAssertCFSource");
            headersList.add("NullAssertCFComment");
            headersList.add("NullAssertCFTarget");

            headersList.add("TrueFalseAssertCFSource");
            headersList.add("TrueFalseAssertCFComment");
            headersList.add("TrueFalseAssertCFTarget");

            headersList.add("ShuffleNamesCFSource");
            headersList.add("ShuffleNamesCFComment");
            headersList.add("ShuffleNamesCFTarget");

            headersList.add("RandomNamesCFSource");
            headersList.add("RandomNamesCFComment");
            headersList.add("RandomNamesCFTarget");


            headersList.add("IndependentSwapCFSource");
            headersList.add("IndependentSwapCFComment");
            headersList.add("IndependentSwapCFTarget");

            headersList.add("defUseBreakCFSource");
            headersList.add("defUseBreakCFComment");
            headersList.add("defUseBreakCFTarget");


            // Writing the updated records with additional columns
            try (CSVPrinter printer = new CSVPrinter(new FileWriter(outputPath), CSVFormat.DEFAULT.withHeader(headersList.toArray(new String[0])))) {
                int count = 0;
                for (CSVRecord record : records) {

                    String TagglesssourceCode = record.get("formatted_source");
                    String taggedSource = record.get("source_code");
                    String targetCode = record.get("formatted_target");
                    String SingletargetCode = record.get("target");
                    String reviewComment = record.get("review_comment");
                    String taggedSubstring = FixTags.extractSubstring(taggedSource);

                    targetCode = targetCode.replaceAll("(?m)^[ \\t]*\\r?\\n|^[ \\t]*$", "");
                    TagglesssourceCode = TagglesssourceCode.replaceAll("(?m)^[ \\t]*\\r?\\n|^[ \\t]*$", "");
                    String sourceCode = SourceCodePreprocessor.processSourceCode(TagglesssourceCode, taggedSource);

                    System.out.println("---------------------------------------------------------------------");
                    System.out.println("Record Number:\n" + count);
                    System.out.println("Formatted Source Code without tag:\n" + TagglesssourceCode);
                    System.out.println("Tagged Source Code:\n" + taggedSource);
                    System.out.println("Tagged formatted Source Code:\n" + sourceCode);
                    System.out.println("needs revision:\n" + taggedSubstring);
                    System.out.println("Review comment:\n" + reviewComment);
                    System.out.println("Formatted Target Code:\n" + targetCode);

                    List<Object> newRecord = new ArrayList<>();
                    record.forEach(newRecord::add);



                    JavaParser javaParser = new JavaParser();
                    ParseResult<CompilationUnit> sourceParseResult = javaParser.parse(sourceCode);
                    ParseResult<CompilationUnit> targetParseResult = javaParser.parse(targetCode);

                    if (sourceParseResult.isSuccessful() & sourceParseResult.getResult().isPresent() & targetParseResult.isSuccessful() & targetParseResult.getResult().isPresent()) {
                        CompilationUnit sourceCu = sourceParseResult.getResult().get();
                        CompilationUnit targetCu = targetParseResult.getResult().get();


                        // Data type change
                        CompilationUnit WrapperdataSourceCu = sourceCu.clone();
                        SafeTypeWrapperTransformer wrapperTransformer = new SafeTypeWrapperTransformer();
                        DataTypeTransformer2 transformer = new DataTypeTransformer2();
                        SafeTypeWrapperTransformer.TransformationResult wrapperResult =
                                wrapperTransformer.transform(WrapperdataSourceCu, taggedSource, reviewComment, SingletargetCode);

                        if (wrapperResult.sourceChanged) {
                            String WrapperSource = wrapperResult.modifiedSource;
                            String WrapperComment = wrapperResult.modifiedComment;
                            String WrapperTarget = wrapperResult.modifiedTarget;

                            DataTypeTransformer2.TransformationResult transresult = transformer.transformCode(WrapperSource, WrapperTarget, WrapperComment);

                            newRecord.add(transresult.modifiedSource);
                            System.out.println("Data type change applied to sourcecode: " + transresult.modifiedSource);

                            newRecord.add(transresult.modifiedComment);
                            System.out.println("Data type change applied to comment: " + transresult.modifiedComment);

                            newRecord.add(transresult.modifiedTarget);
                            System.out.println("Data type change applied to target: " + transresult.modifiedTarget);

                        } else {
                            DataTypeTransformer2.TransformationResult transresult2 = transformer.transformCode(taggedSource, SingletargetCode, reviewComment);

                            if (transresult2.sourceChanged) {
                                newRecord.add(transresult2.modifiedSource);
                                System.out.println("Data type change applied to sourcecode: " + transresult2.modifiedSource);

                                newRecord.add(transresult2.modifiedComment);
                                System.out.println("Data type change applied to comment: " + transresult2.modifiedComment);

                                newRecord.add(transresult2.modifiedTarget);
                                System.out.println("Data type change applied to target: " + transresult2.modifiedTarget);

                            } else {
                                newRecord.add("NA");
                                newRecord.add("NA");
                                newRecord.add("NA");
                            }
                        }


                        //If-Else Swap
                        CompilationUnit ifSourceCu = sourceCu.clone();
                        CompilationUnit ifTargetCu = targetCu.clone();
                        IfManipulator ifManipulator = new IfManipulator();
                        SingleIfManipulator singleIfManipulator = new SingleIfManipulator();
                        if (ifManipulator.containsIfElse(ifSourceCu)) {
                            ifSourceCu.accept(ifManipulator, null);
                        } else {
                            ifSourceCu.accept(singleIfManipulator, null);
                        }
                        String modifiedifCode = ifSourceCu.toString();
                        modifiedifCode = modifiedifCode.replaceAll("(?m)^[ \\t]*\\r?\\n|^[ \\t]*$", "");
                        boolean anyModificationif = ifManipulator.isModified() || singleIfManipulator.SingleisModified();
                        if (anyModificationif) {
                            System.out.println("If-Else Swap Applied to source code:\n" + modifiedifCode);
                            String formattedModifiedifCode = CodeFormatter.convertTagsWithOrder(CodeFormatter.formatCode(modifiedifCode));
                            System.out.println("**Formatted|If-Else Swap Applied to source code:\n" + formattedModifiedifCode);
                            newRecord.add(formattedModifiedifCode);
                            newRecord.add(reviewComment);
                        } else {
                            System.out.println("No If-Else swap applied to the source code.");
                            newRecord.add("NA");
                            newRecord.add("NA");
                            newRecord.add("NA");
                        }
                        if (ifManipulator.containsIfElse(ifTargetCu)) {
                            ifTargetCu.accept(ifManipulator, null);
                        } else {
                            ifTargetCu.accept(singleIfManipulator, null);
                        }
                        String modifiedTargetifCode = ifTargetCu.toString();
                        modifiedTargetifCode = modifiedTargetifCode.replaceAll("(?m)^[ \\t]*\\r?\\n|^[ \\t]*$", "");
                        boolean targetAnyModificationif = ifManipulator.isModified() || singleIfManipulator.SingleisModified();
                        if (anyModificationif & targetAnyModificationif) {
                            System.out.println("If-Else Swap Applied to target code:\n" + modifiedTargetifCode);
                            String formattedModifiedTaragetifCode = CodeFormatter.formatCode(modifiedTargetifCode);
                            System.out.println("**Formatted|If-Else Swap Applied to target code:\n" + formattedModifiedTaragetifCode);
                            newRecord.add(formattedModifiedTaragetifCode);

                        } if (anyModificationif & !targetAnyModificationif) {
                            System.out.println("No If-Else swap applied to the target code.");
                            newRecord.add(SingletargetCode);
                        }


                        //Exception adder
                        CompilationUnit exceptionSourceCu = sourceCu.clone();
                        CompilationUnit exceptionTargetCu = targetCu.clone();
                        ExceptionAdder exceptionAdder = new ExceptionAdder();
                        exceptionSourceCu.accept(exceptionAdder, null);
                        boolean ExceptionSourceModified = exceptionAdder.isModified();
                        if (ExceptionSourceModified) {
                            String modifiedExceptionCode = exceptionSourceCu.toString();
                            modifiedExceptionCode = modifiedExceptionCode.replaceAll("(?m)^[ \\t]*\\r?\\n|^[ \\t]*$", "");

                            FixTags.MatchResult Exceptiontagresult = FixTags.matchSubstring(modifiedExceptionCode, taggedSubstring, null, null);
                            if (Exceptiontagresult.found)
                                modifiedExceptionCode = CodeFormatter.formatCode(Exceptiontagresult.modifiedTarget);
                            else
                                modifiedExceptionCode = CodeFormatter.convertTagsWithOrder(CodeFormatter.formatCode(modifiedExceptionCode));
                            System.out.println("Add Exception to source code:\n" + modifiedExceptionCode);
                            newRecord.add(modifiedExceptionCode);

                            newRecord.add(reviewComment);
                        } else {
                            System.out.println("No Exception added to the source code.");
                            newRecord.add("NA");
                            newRecord.add("NA");
                            newRecord.add("NA");
                        }
                        exceptionTargetCu.accept(exceptionAdder, null);
                        boolean ExceptiontargetModified = exceptionAdder.isModified();
                        if (ExceptionSourceModified & ExceptiontargetModified) {
                            String modifiedTargetExceptionCode = exceptionTargetCu.toString();
                            modifiedTargetExceptionCode = modifiedTargetExceptionCode.replaceAll("(?m)^[ \\t]*\\r?\\n|^[ \\t]*$", "");
                            String formattedModifiedtargetExceptionCode = CodeFormatter.formatCode(modifiedTargetExceptionCode);
                            System.out.println("Add Exception to target code:\n" + modifiedTargetExceptionCode);
                            newRecord.add(formattedModifiedtargetExceptionCode);
                        }if (ExceptionSourceModified & !ExceptiontargetModified) {
                            System.out.println("No Exception added to the target code.");
                            newRecord.add(SingletargetCode);
                        }

                        //Dead code
                        CompilationUnit deadCodeSourceCu = sourceCu.clone();
                        CompilationUnit deadCodeTargetCu = targetCu.clone();
                        DeadCodeAdder deadCodeAdder = new DeadCodeAdder();
                        deadCodeSourceCu.accept(deadCodeAdder, null);
                        boolean deadCodeSourceModified = deadCodeAdder.isModified();
                        if (deadCodeSourceModified) {
                            String modifiedDeadCodeSource = deadCodeSourceCu.toString();
                            modifiedDeadCodeSource = modifiedDeadCodeSource.replaceAll("(?m)^[ \\t]*\\r?\\n|^[ \\t]*$", "");
                            FixTags.MatchResult DeadCodetagresult = FixTags.matchSubstring(modifiedDeadCodeSource, taggedSubstring, null, null);
                            if (DeadCodetagresult.found)
                                modifiedDeadCodeSource = CodeFormatter.formatCode(DeadCodetagresult.modifiedTarget);
                            else
                                modifiedDeadCodeSource = CodeFormatter.convertTagsWithOrder(CodeFormatter.formatCode(modifiedDeadCodeSource));
                            System.out.println("Dead code added to source code:\n" + modifiedDeadCodeSource);
                            newRecord.add(modifiedDeadCodeSource);
                            newRecord.add(reviewComment);
                        } else {
                            System.out.println("No dead code added to the source code.");
                            newRecord.add("NA");
                            newRecord.add("NA");
                            newRecord.add("NA");
                        }
                        deadCodeTargetCu.accept(deadCodeAdder, null);
                        boolean deadCodeTargetModified = deadCodeAdder.isModified();
                        if (deadCodeSourceModified & deadCodeTargetModified) {
                            String modifiedDeadCodeTarget = deadCodeTargetCu.toString();
                            modifiedDeadCodeTarget = modifiedDeadCodeTarget.replaceAll("(?m)^[ \\t]*\\r?\\n|^[ \\t]*$", "");
                            String formattedModifiedDeadCodeTarget = CodeFormatter.formatCode(modifiedDeadCodeTarget);
                            System.out.println("Dead code added to target code:\n" + modifiedDeadCodeTarget);
                            newRecord.add(formattedModifiedDeadCodeTarget);
                        } if (deadCodeSourceModified & !deadCodeTargetModified) {
                            System.out.println("No dead code added to the target code.");
                            newRecord.add(SingletargetCode);
                        }




                        // Try-Catch Modifier

                        CompilationUnit tryCatchSourceCu = sourceCu.clone();
                        CompilationUnit tryCatchTargetCu = targetCu.clone();
                        TryCatchModifier tryCatchModifier = new TryCatchModifier();

                        tryCatchSourceCu.accept(tryCatchModifier, null);
                        boolean tryCatchSourceModified = tryCatchModifier.isModified();
                        if (tryCatchSourceModified) {
                            String modifiedTryCatchCode = tryCatchSourceCu.toString().replaceAll("(?m)^[ \\t]*\\r?\\n|^[ \\t]*$", "");
                            FixTags.MatchResult TryCatchtagresult = FixTags.matchSubstring(modifiedTryCatchCode, taggedSubstring, null, null);
                            if (TryCatchtagresult.found)
                                modifiedTryCatchCode = CodeFormatter.formatCode(TryCatchtagresult.modifiedTarget);
                            else
                                modifiedTryCatchCode = CodeFormatter.convertTagsWithOrder(CodeFormatter.formatCode(modifiedTryCatchCode));
                            System.out.println("Try and Catch Added to source code:\n" + modifiedTryCatchCode);
                            newRecord.add(modifiedTryCatchCode);
                            newRecord.add(reviewComment);
                        } else {
                            System.out.println("No Try and Catch added to source code.");
                            newRecord.add("NA");
                            newRecord.add("NA");
                            newRecord.add("NA");
                        }

                        tryCatchTargetCu.accept(tryCatchModifier, null);
                        boolean tryCatchTargetModified = tryCatchModifier.isModified();
                        if (tryCatchSourceModified & tryCatchTargetModified) {
                            String modifiedTargetTryCatchCode = tryCatchTargetCu.toString().replaceAll("(?m)^[ \\t]*\\r?\\n|^[ \\t]*$", "");
                            String formattedModifiedTargetTryCatchCode = CodeFormatter.formatCode(modifiedTargetTryCatchCode);
                            System.out.println("Try and Catch Added to target code:\n" + modifiedTargetTryCatchCode);
                            newRecord.add(formattedModifiedTargetTryCatchCode);
                        } if (tryCatchSourceModified & !tryCatchTargetModified) {
                            System.out.println("No Try and Catch added to the target code.");
                            newRecord.add(SingletargetCode);
                        }

                        // DataFlow Modifier
                        CompilationUnit DataFlowSourceCu = sourceCu.clone();
                        CompilationUnit DataFlowTargetCu = targetCu.clone();
                        DataFlowChange DataFlowModifier = new DataFlowChange();
                        DataFlowChangeT DataFlowModifierT = new DataFlowChangeT();
                        DataFlowSourceCu.accept(DataFlowModifier, null);
                        boolean DataFlowSourceModified = DataFlowModifier.isModified();
                        if (DataFlowSourceModified) {
                            DataFlowTargetCu.accept(DataFlowModifierT, null);
                            String DFmodifiedSourceCodee = DataFlowSourceCu.toString().replaceAll("(?m)^[ \\t]*\\r?\\n|^[ \\t]*$", "");
                            FixTags.MatchResult DataFlowtagresult = FixTags.matchSubstring(DFmodifiedSourceCodee, taggedSubstring, null, null);
                            if (DataFlowtagresult.found)
                                DFmodifiedSourceCodee = CodeFormatter.formatCode(DataFlowtagresult.modifiedTarget);
                            else
                                DFmodifiedSourceCodee = CodeFormatter.convertTagsWithOrder(CodeFormatter.formatCode(DFmodifiedSourceCodee));
                            System.out.println("Data flow changed in source code:\n" + DFmodifiedSourceCodee);
                            newRecord.add(DFmodifiedSourceCodee);
                            newRecord.add(reviewComment);
                        } else {
                            System.out.println("No change of data flow in source code.");
                            newRecord.add("NA");
                            newRecord.add("NA");
                            newRecord.add("NA");
                        }


                        boolean DataFlowTargetModified = DataFlowModifier.isModified();
                        if (DataFlowSourceModified & DataFlowTargetModified) {
                            String DFmodifiedTargetCode = DataFlowTargetCu.toString().replaceAll("(?m)^[ \\t]*\\r?\\n|^[ \\t]*$", "");
                            String formattedDFModifiedTargetCode = CodeFormatter.formatCode(DFmodifiedTargetCode);
                            System.out.println("Data flow changed in target code:\n" + DFmodifiedTargetCode);
                            newRecord.add(formattedDFModifiedTargetCode);
                        } if (DataFlowSourceModified & !DataFlowTargetModified) {
                            System.out.println("No change of data flow in the target code.");
                            newRecord.add(SingletargetCode);
                        }

                        // Null Assert converted to If-Else

                        CompilationUnit assertSourceCu = sourceCu.clone();
                        CompilationUnit assertTargetCu = targetCu.clone();
                        NotNullAssertConverter nullconverter = new NotNullAssertConverter();

                        assertSourceCu.accept(nullconverter, null);
                        boolean assertSourceModified = nullconverter.isModified();
                        if (assertSourceModified) {
                            String modifiedAssertSourceCode = assertSourceCu.toString().replaceAll("(?m)^[ \\t]*\\r?\\n|^[ \\t]*$", "");
                            FixTags.MatchResult NullAsserttagresult = FixTags.matchSubstring(modifiedAssertSourceCode, taggedSubstring, null, null);
                            if (NullAsserttagresult.found)
                                modifiedAssertSourceCode = CodeFormatter.formatCode(NullAsserttagresult.modifiedTarget);
                            else
                                modifiedAssertSourceCode = CodeFormatter.convertTagsWithOrder(CodeFormatter.formatCode(modifiedAssertSourceCode));
                            System.out.println("Null Assert converted to If-Else in source:\n" + modifiedAssertSourceCode);
                            newRecord.add(modifiedAssertSourceCode);
                            newRecord.add(reviewComment);
                        } else {
                            System.out.println("No null assert convert applied to source.");
                            newRecord.add("NA");
                            newRecord.add("NA");
                            newRecord.add("NA");
                        }
                        assertTargetCu.accept(nullconverter, null);
                        boolean assertTargetModified = nullconverter.isModified();
                        if (assertSourceModified & assertTargetModified) {
                            String modifiedAssertTargetCode = assertTargetCu.toString().replaceAll("(?m)^[ \\t]*\\r?\\n|^[ \\t]*$", "");
                            String formattedAssertTargetCode = CodeFormatter.formatCode(modifiedAssertTargetCode);
                            System.out.println("Null Assert converted to If-Else in target:\n" + modifiedAssertTargetCode);
                            newRecord.add(formattedAssertTargetCode);
                        } if (assertSourceModified & !assertTargetModified) {
                            System.out.println("No null assert convert applied to target.");
                            newRecord.add(SingletargetCode);
                        }


                        // Equal Assert converted to If-Else
                        CompilationUnit equalAssertSourceCu = sourceCu.clone();
                        CompilationUnit equalAssertTargetCu = targetCu.clone();
                        EqualsAssertConverter Equalconverter = new EqualsAssertConverter();

                        equalAssertSourceCu.accept(Equalconverter, null);
                        boolean equalAssertSourceModified = Equalconverter.isModified();
                        if (equalAssertSourceModified) {
                            String modifiedEqualAssertSourceCode = equalAssertSourceCu.toString().replaceAll("(?m)^[ \\t]*\\r?\\n|^[ \\t]*$", "");
                            FixTags.MatchResult EqualAsserttagresult = FixTags.matchSubstring(modifiedEqualAssertSourceCode, taggedSubstring, null, null);
                            if (EqualAsserttagresult.found)
                                modifiedEqualAssertSourceCode = CodeFormatter.formatCode(EqualAsserttagresult.modifiedTarget);
                            else
                                modifiedEqualAssertSourceCode = CodeFormatter.convertTagsWithOrder(CodeFormatter.formatCode(modifiedEqualAssertSourceCode));
                            System.out.println("Equal Assert converted to If-Else in source:\n" + modifiedEqualAssertSourceCode);
                            newRecord.add(modifiedEqualAssertSourceCode);
                            newRecord.add(reviewComment);
                        } else {
                            System.out.println("No equal assert convert applied to source.");
                            newRecord.add("NA");
                            newRecord.add("NA");
                            newRecord.add("NA");
                        }

                        equalAssertTargetCu.accept(Equalconverter, null);
                        boolean equalAssertTargetModified = Equalconverter.isModified();
                        if (equalAssertSourceModified & equalAssertTargetModified) {
                            String modifiedEqualAssertTargetCode = equalAssertTargetCu.toString().replaceAll("(?m)^[ \\t]*\\r?\\n|^[ \\t]*$", "");
                            String formattedEqualAssertTargetCode = CodeFormatter.formatCode(modifiedEqualAssertTargetCode);
                            System.out.println("Equal Assert converted to If-Else in target:\n" + modifiedEqualAssertTargetCode);
                            newRecord.add(formattedEqualAssertTargetCode);
                        } if (equalAssertSourceModified & !equalAssertTargetModified) {
                            System.out.println("No equal assert convert applied to target.");
                            newRecord.add(SingletargetCode);
                        }

                        // True/False Assert converted to If-Else
                        CompilationUnit booleanAssertSourceCu = sourceCu.clone();
                        CompilationUnit booleanAssertTargetCu = targetCu.clone();
                        BooleanAssertConverter TFconverter = new BooleanAssertConverter();
                        booleanAssertSourceCu.accept(TFconverter, null);
                        boolean booleanAssertSourceModified = TFconverter.isModified();
                        if (booleanAssertSourceModified) {
                            String modifiedBooleanAssertSourceCode = booleanAssertSourceCu.toString().replaceAll("(?m)^[ \\t]*\\r?\\n|^[ \\t]*$", "");

                            FixTags.MatchResult BooleanAsserttagresult = FixTags.matchSubstring(modifiedBooleanAssertSourceCode, taggedSubstring, null, null);
                            if (BooleanAsserttagresult.found)
                                modifiedBooleanAssertSourceCode = CodeFormatter.formatCode(BooleanAsserttagresult.modifiedTarget);
                            else
                                modifiedBooleanAssertSourceCode = CodeFormatter.convertTagsWithOrder(CodeFormatter.formatCode(modifiedBooleanAssertSourceCode));

                            System.out.println("True/False Assert converted to If-Else in source:\n" + modifiedBooleanAssertSourceCode);
                            newRecord.add(modifiedBooleanAssertSourceCode);
                            newRecord.add(reviewComment);
                        } else {
                            System.out.println("No True/false assert convert applied to source.");
                            newRecord.add("NA");
                            newRecord.add("NA");
                            newRecord.add("NA");
                        }
                        booleanAssertTargetCu.accept(TFconverter, null);
                        boolean booleanAssertTargetModified = TFconverter.isModified();
                        if (booleanAssertSourceModified & booleanAssertTargetModified) {
                            String modifiedBooleanAssertTargetCode = booleanAssertTargetCu.toString().replaceAll("(?m)^[ \\t]*\\r?\\n|^[ \\t]*$", "");
                            String formattedBooleanAssertTargetCode = CodeFormatter.formatCode(modifiedBooleanAssertTargetCode);
                            System.out.println("True/False Assert converted to If-Else in target:\n" + modifiedBooleanAssertTargetCode);
                            newRecord.add(formattedBooleanAssertTargetCode);
                        } if (booleanAssertSourceModified & !booleanAssertTargetModified) {
                            System.out.println("No True/false assert convert applied to target.");
                            newRecord.add(SingletargetCode);
                        }


                        // Shuffle Variable Names
                        CompilationUnit sourceCuClone = sourceCu.clone();
                        CompilationUnit targetCuClone = targetCu.clone();
                        VariableShuffler shuffler = new VariableShuffler();
                        String modifiedComment = shuffler.shuffleVariables(sourceCuClone, targetCuClone, reviewComment);
                        boolean variablesShuffled = shuffler.isModified();
                        if (variablesShuffled) {
                            System.out.println("Variable name mapping:");
                            shuffler.getNameMapping().forEach((original, newName) -> System.out.println(original + " -> " + newName));
                            System.out.println("Modified Review Comment: " + modifiedComment);

                            String formattedSourceCode = CodeFormatter.convertTagsWithOrder(CodeFormatter.formatCode(sourceCuClone.toString().replaceAll("(?m)^[ \\t]*\\r?\\n|^[ \\t]*$", "")));
                            String formattedTargetCode = CodeFormatter.formatCode(targetCuClone.toString().replaceAll("(?m)^[ \\t]*\\r?\\n|^[ \\t]*$", ""));
                            System.out.println("Variable names shuffled in the source Code:\n" + formattedSourceCode);
                            System.out.println("Variable names shuffled in the target Code:\n" + formattedTargetCode);

                            newRecord.add(formattedSourceCode);
                            newRecord.add(modifiedComment);
                            newRecord.add(formattedTargetCode);
                        } else {
                            System.out.println("No variable name shuffle for source code and target code.");
                            newRecord.add("NA");
                            newRecord.add("NA");
                            newRecord.add("NA");
                        }


                        // convert variable names to random strings
                        CompilationUnit randomSourceCu = sourceCu.clone();
                        CompilationUnit randomTargetCu = targetCu.clone();
                        RandomNameAssigner nameAssigner = new RandomNameAssigner();
                        String randomModifiedComment = nameAssigner.assignRandomNames(randomSourceCu, randomTargetCu, reviewComment);

                        boolean namesRandomlyAssigned = nameAssigner.isModified();
                        if (namesRandomlyAssigned) {
                            System.out.println("Random variable name mapping:");
                            nameAssigner.getNameMapping().forEach((original, newName) -> System.out.println(original + " -> " + newName));
                            System.out.println("Modified Review Comment: " + randomModifiedComment);

                            String formattedRandomSourceCode = CodeFormatter.convertTagsWithOrder(CodeFormatter.formatCode(randomSourceCu.toString().replaceAll("(?m)^[ \\t]*\\r?\\n|^[ \\t]*$", "")));
                            String formattedRandomTargetCode = CodeFormatter.formatCode(randomTargetCu.toString().replaceAll("(?m)^[ \\t]*\\r?\\n|^[ \\t]*$", ""));

                            System.out.println("Random names assigned in the source Code:\n" + formattedRandomSourceCode);
                            System.out.println("Random names assigned in the target Code:\n" + formattedRandomTargetCode);

                            newRecord.add(formattedRandomSourceCode);
                            newRecord.add(randomModifiedComment);
                            newRecord.add(formattedRandomTargetCode);
                        } else {
                            System.out.println("No random name assignment for source code and target code.");
                            newRecord.add("NA");
                            newRecord.add("NA");
                            newRecord.add("NA");
                        }


                        // Independent line swap (IndLineSwap)
                        CompilationUnit IndLineSwapSourceCu = sourceCu.clone();
                        CompilationUnit IndLineSwapTargetCu = targetCu.clone();
                        IndependentLineSwap independentLineSwap = new IndependentLineSwap();

                        IndLineSwapSourceCu.accept(independentLineSwap, null);
                        boolean IndLineSwapModified = independentLineSwap.isModified();
                        if (IndLineSwapModified) {
                            String modifiedIndLineSwapSourceCode = IndLineSwapSourceCu.toString().replaceAll("(?m)^[ \\t]*\\r?\\n|^[ \\t]*$", "");

                            FixTags.MatchResult IndSwaptagresult = FixTags.matchSubstring(modifiedIndLineSwapSourceCode, taggedSubstring, null, null);
                            if (IndSwaptagresult.found)
                                modifiedIndLineSwapSourceCode = CodeFormatter.formatCode(IndSwaptagresult.modifiedTarget);
                            else
                                modifiedIndLineSwapSourceCode = CodeFormatter.convertTagsWithOrder(CodeFormatter.formatCode(modifiedIndLineSwapSourceCode));

                            System.out.println("Independent lines swapped in source:\n" + modifiedIndLineSwapSourceCode);


                            newRecord.add(modifiedIndLineSwapSourceCode);
                            newRecord.add(reviewComment);
                        } else {
                            System.out.println("No independent line swapped in source.");
                            newRecord.add("NA");
                            newRecord.add("NA");
                            newRecord.add("NA");
                        }
                        IndLineSwapTargetCu.accept(independentLineSwap, null);
                        boolean IndLineSwapTargetModified = independentLineSwap.isModified();
                        if (IndLineSwapModified & IndLineSwapTargetModified) {

                            String modifiedIndLineSwapTargetCode = IndLineSwapTargetCu.toString().replaceAll("(?m)^[ \\t]*\\r?\\n|^[ \\t]*$", "");
                            String formattedIndLineSwapTargetCode = CodeFormatter.formatCode(modifiedIndLineSwapTargetCode);
                            System.out.println("Independent lines swapped in in target:\n" + modifiedIndLineSwapTargetCode);
                            newRecord.add(formattedIndLineSwapTargetCode);
                        }
                        if (IndLineSwapModified & !IndLineSwapTargetModified) {
                            System.out.println("No independent line swapped in target.");
                            newRecord.add(SingletargetCode);
                        }

                        // Def-Use Break Mutation
                        CompilationUnit defUseBreakSourceCu = sourceCu.clone();
                        CompilationUnit defUseBreakTargetCu = targetCu.clone();
                        DefUseBreakMutator defUseBreakMutator = new DefUseBreakMutator();

                        defUseBreakSourceCu.accept(defUseBreakMutator, null);
                        boolean defUseBreakSourceModified = defUseBreakMutator.isModified();
                        String defUseBreakSourceRandomName = defUseBreakMutator.getLastRandomNameUsed();
                        String OriginalName = defUseBreakMutator.OrgName();

                        if (defUseBreakSourceModified) {
                            String modifiedDefUseBreakSourceCode = defUseBreakSourceCu.toString().replaceAll("(?m)^[ \\t]*\\r?\\n|^[ \\t]*$", "");

                            FixTags.MatchResult DefUsetagresult = FixTags.matchSubstring(modifiedDefUseBreakSourceCode, taggedSubstring, defUseBreakSourceRandomName, OriginalName);
                            if (DefUsetagresult.found)
                                modifiedDefUseBreakSourceCode = CodeFormatter.formatCode(DefUsetagresult.modifiedTarget);
                            else
                                modifiedDefUseBreakSourceCode = CodeFormatter.convertTagsWithOrder(CodeFormatter.formatCode(modifiedDefUseBreakSourceCode));

                            System.out.println("Def-Use chain broken in source:\n" + modifiedDefUseBreakSourceCode);
                            newRecord.add(modifiedDefUseBreakSourceCode);
                            newRecord.add(reviewComment);
                        } else {
                            System.out.println("No def-use chain broke in source.");
                            newRecord.add("NA");
                            newRecord.add("NA");
                            newRecord.add("NA");
                        }

                        // Apply to target
                        DefUseBreakMutator targetMutator = new DefUseBreakMutator(defUseBreakSourceRandomName);
                        defUseBreakTargetCu.accept(targetMutator, null);
                        boolean defUseBreakTargetModified = targetMutator.isModified();
                        if (defUseBreakSourceModified && defUseBreakTargetModified) {
                            String modifiedDefUseBreakTargetCode = defUseBreakTargetCu.toString().replaceAll("(?m)^[ \\t]*\\r?\\n|^[ \\t]*$", "");
                            String formattedDefUseBreakTargetCode = CodeFormatter.formatCode(modifiedDefUseBreakTargetCode);
                            System.out.println("Def-Use chain broken in target:\n" + modifiedDefUseBreakTargetCode);
                            newRecord.add(formattedDefUseBreakTargetCode);
                        }
                        if (defUseBreakSourceModified && !defUseBreakTargetModified) {
                            System.out.println("No def-use chain broke in target.");
                            newRecord.add(SingletargetCode);
                        }

                    }

                    printer.printRecord(newRecord);
                    count++;
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
