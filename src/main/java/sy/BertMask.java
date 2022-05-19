package sy;

import ai.onnxruntime.*;
import org.apache.commons.lang3.tuple.Triple;
import sy.bert.LoadModel;
import sy.bert.tokenizerimpl.BertTokenizer;
import util.CollectionUtil;

import java.util.List;
import java.util.Map;

/**
 * @author sy
 * @date 2022/5/2 14:03
 */
public class BertMask {
    static BertTokenizer tokenizer;
    public static void main(String[] args) {
        String text = "中国的首都是北[MASK]。";
        text = "我家后面有一[MASK]大树。";
        Triple<BertTokenizer, Map<String, OnnxTensor>, Integer> triple = null;
        try {
            triple = parseInputText(text);
        } catch (Exception e) {
            e.printStackTrace();
        }
        var maskPredictions = predMask(triple);
        System.out.println(maskPredictions);
    }

    static {
        tokenizer = new BertTokenizer();
        try {
            LoadModel.loadOnnxModel();
        } catch (OrtException e) {
            e.printStackTrace();
        }
    }

    /**
     * tokenize text
     * @param text
     * @return
     * @throws Exception
     */
    public static Triple<BertTokenizer, Map<String, OnnxTensor>, Integer> parseInputText(String text) throws Exception{
        var env = LoadModel.env;
        List<String > tokens = tokenizer.tokenize(text);

        System.out.println(tokens);

        List<Integer> tokenIds = tokenizer.convert_tokens_to_ids(tokens);
        int maskId = tokenIds.indexOf(tokenizer.convert_tokens_to_ids("[MASK]"));
        long[] inputIds = new long[tokenIds.size()];
        long[] attentionMask = new long[tokenIds.size()];
        long[] tokenTypeIds = new long[tokenIds.size()];
        for(int index=0; index < tokenIds.size(); index ++) {
            inputIds[index] = tokenIds.get(index);
            attentionMask[index] = 1;
            tokenTypeIds[index] = 0;
        }
        long[] shape = new long[]{1, inputIds.length};
        Object ObjInputIds = OrtUtil.reshape(inputIds, shape);
        Object ObjAttentionMask = OrtUtil.reshape(attentionMask, shape);
        Object ObjTokenTypeIds = OrtUtil.reshape(tokenTypeIds, shape);
        OnnxTensor input_ids = OnnxTensor.createTensor(env, ObjInputIds);
        OnnxTensor attention_mask = OnnxTensor.createTensor(env, ObjAttentionMask);
        OnnxTensor token_type_ids = OnnxTensor.createTensor(env, ObjTokenTypeIds);
        var inputs = Map.of("input_ids", input_ids, "attention_mask", attention_mask, "token_type_ids", token_type_ids);
        return Triple.of(tokenizer, inputs, maskId);
    }

    /**
     * predict mask
     * @param triple
     * @return
     */
    public static List<String> predMask(Triple<BertTokenizer, Map<String, OnnxTensor>, Integer> triple) {
        return predMask(triple, 5);
    }

    public static List<String> predMask(Triple<BertTokenizer, Map<String, OnnxTensor>, Integer> triple, int topK) {
        var tokenizer = triple.getLeft();
        var inputs =triple.getMiddle();
        var maskId = triple.getRight();
        List<String> maskResults = null;
        try{
            var session = LoadModel.session;
            try(var results = session.run(inputs)) {
                OnnxValue onnxValue = results.get(0);
                float[][][] labels = (float[][][]) onnxValue.getValue();
                float[] maskLables = labels[0][maskId];
                int[] index = predSort(maskLables);
                maskResults = CollectionUtil.newArrayList();
                for(int idx=0; idx < topK; idx ++) {
                    maskResults.add(tokenizer.convert_ids_to_tokens(index[idx]));
                }
            }
        } catch (OrtException e) {
            e.printStackTrace();
        }
        return maskResults;
    }

    /**
     * 得到最大概率label对应的index
     * @param probabilities
     * @return
     */
    public static int predMax(float[] probabilities) {
        float maxVal = Float.NEGATIVE_INFINITY;
        int idx = 0;
        for (int i = 0; i < probabilities.length; i++) {
            if (probabilities[i] > maxVal) {
                maxVal = probabilities[i];
                idx = i;
            }
        }
        return idx;
    }

    /**
     * 对预测的概率进行排序
     * @param probabilities
     * @return
     */
    private static int[] predSort(float[] probabilities) {
        int[] indices = new int[probabilities.length];
        for (int i = 0; i < probabilities.length; i++) {
            indices[i] = i;
        }
        predSort(probabilities, 0, probabilities.length-1, indices);
        return indices;
    }

    private static void predSort(float[] probabilities, int begin, int end, int[] indices) {
        if (begin >= 0 && begin < probabilities.length && end >= 0 && end < probabilities.length && begin < end) {
            int i = begin, j = end;
            float vot = probabilities[i];
            int temp = indices[i];
            while (i != j) {
                while(i < j && probabilities[j] <= vot) j--;
                if(i < j) {
                    probabilities[i] = probabilities[j];
                    indices[i] = indices[j];
                    i++;
                }
                while(i < j && probabilities[i] >= vot)  i++;
                if(i < j) {
                    probabilities[j] = probabilities[i];
                    indices[j] = indices[i];
                    j--;
                }
            }
            probabilities[i] = vot;
            indices[i] = temp;
            predSort(probabilities, begin, j-1, indices);
            predSort(probabilities, i+1, end, indices);
        }
    }

}

