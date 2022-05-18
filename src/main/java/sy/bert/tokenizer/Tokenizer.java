package sy.bert.tokenizer;

import java.util.List;

/**
 * @author sy
 * @date 2022/5/2 14:03
 */
public interface Tokenizer {
	public List<String> tokenize(String text);
}
