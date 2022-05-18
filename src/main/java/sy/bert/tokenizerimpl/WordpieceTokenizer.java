package sy.bert.tokenizerimpl;

import sy.bert.tokenizer.Tokenizer;
import sy.bert.utils.TokenizerUtils;
import util.CollectionUtil;

import java.util.List;
import java.util.Map;

/**
 * @author sy
 * @date 2022/5/2 14:03
 */
public class WordpieceTokenizer implements Tokenizer {
	private Map<String, Integer> vocab;
	private String unk_token;
	private int max_input_chars_per_word;

	public WordpieceTokenizer(Map<String, Integer> vocab, String unk_token, int max_input_chars_per_word) {
		this.vocab = vocab;
		this.unk_token = unk_token;
		this.max_input_chars_per_word = max_input_chars_per_word;
	}

	public WordpieceTokenizer(Map<String, Integer> vocab, String unk_token) {
		this.vocab = vocab;
		this.unk_token = unk_token;
		this.max_input_chars_per_word = 100;
	}

	@Override
	public List<String> tokenize(String text) {
		List<String> output_tokens = CollectionUtil.newArrayList();
		for (String token : TokenizerUtils.whitespace_tokenize(text)) {
			if (token.length() > max_input_chars_per_word) {
				output_tokens.add(unk_token);
				continue;
			}
			boolean is_bad = false;
			int start = 0;

			List<String> sub_tokens = CollectionUtil.newArrayList();
			while (start < token.length()) {
				int end = token.length();
				String cur_substr = "";
				while (start < end) {
					String substr = token.substring(start, end);
					if (start > 0) {
						substr = "##" + substr;
					}
					if (vocab.containsKey(substr)) {
						cur_substr = substr;
						break;
					}
					end -= 1;
				}
				if (cur_substr == "") {
					is_bad = true;
					break;
				}
				sub_tokens.add(cur_substr);
				start = end;
			}
			if (is_bad) {
				output_tokens.add(unk_token);
			} else {
				output_tokens.addAll(sub_tokens);
			}
		}
		return output_tokens;
	}
}
