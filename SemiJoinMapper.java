import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import java.io.*;
import java.net.URI;
import java.util.StringTokenizer;
import com.google.common.hash.BloomFilter;
import com.google.common.hash.Funnels;

public class SemiJoinMapper extends Mapper<LongWritable, Text, Text, Text> {
    private BloomFilter<String> bloomFilter = BloomFilter.create(Funnels.unencodedCharsFunnel(), 10000, 0.01);

    @Override
    protected void setup(Context context) throws IOException {
        URI[] cacheFiles = context.getCacheFiles();
        if (cacheFiles != null && cacheFiles.length > 0) {
            try (BufferedReader reader = new BufferedReader(new FileReader(new File("relation_R.txt")))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    bloomFilter.put(line.trim());
                }
            }
        }
    }

    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        StringTokenizer tokenizer = new StringTokenizer(value.toString(), "\t");
        if (tokenizer.countTokens() < 2) return;

        String userId = tokenizer.nextToken();
        String transactionAmount = tokenizer.nextToken();

        if (bloomFilter.mightContain(userId)) {
            context.write(new Text(userId), new Text(transactionAmount));
        }
    }
}
