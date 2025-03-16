import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.BitSet;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class SemiJoinBloomFilter implements Tool {

    private Configuration conf;

    @Override
    public void setConf(Configuration conf) {
        this.conf = conf;
    }

    @Override
    public Configuration getConf() {
        return conf;
    }

    public static class SemiJoinMapper extends Mapper<LongWritable, Text, Text, Text> {

        private BitSet bloomFilter;
        private int numHashes;
        private int numBits;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            URI[] cacheFiles = context.getCacheFiles();
            if (cacheFiles != null && cacheFiles.length > 0) {
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                        new FileInputStream(cacheFiles[0].getPath().getName())))) {
                    String line = reader.readLine();
                    if (line != null) {
                        String[] parts = line.split(",");
                        numBits = Integer.parseInt(parts[0]);
                        numHashes = Integer.parseInt(parts[1]);
                        bloomFilter = BitSet.valueOf(fromHexString(parts[2]));
                    }
                }
            } else {
                 throw new IOException("Bloom filter file not found in Distributed Cache.");
            }
        }

        @Override
        protected void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            String line = value.toString();
            String[] parts = line.split("\t");
            if (parts.length >= 2) {
                String joinKey = parts[0];
                if (bloomFilterContains(joinKey)) {
                    context.write(new Text(joinKey), new Text(parts[1]));
                }
            }
        }

        private boolean bloomFilterContains(String key) {
            for (int i = 0; i < numHashes; i++) {
                int index = hash(key + i, numBits);
                if (!bloomFilter.get(index)) {
                    return false;
                }
            }
            return true;
        }

        private int hash(String key, int numBits) {
            return Math.abs(key.hashCode() % numBits);
        }

        private static byte[] fromHexString(String s) {
            int len = s.length();
            byte[] data = new byte[len / 2];
            for (int i = 0; i < len; i += 2) {
                data[i / 2] = (byte) ((Character.digit(s.charAt(i), 16) << 4)
                        + Character.digit(s.charAt(i + 1), 16));
            }
            return data;
        }

    } public static class SemiJoinReducer extends Reducer<Text, Text, Text, Text> {
        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            for (Text value : values) {
                context.write(key, value);
            }
        }
    }
    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new Configuration(), new SemiJoinBloomFilter(), args);
        System.exit(res);
    }
    @Override
    public int run(String[] args) throws Exception {
        if (args.length != 3) {
            System.err.println("Usage: SemiJoinBloomFilter <in> <bloomFilterFile> <out>");
            return 2;
        }

        Configuration conf = getConf();
        Job job = Job.getInstance(conf, "Semi-Join with Bloom Filter");
        job.setJarByClass(SemiJoinBloomFilter.class);

        job.addCacheFile(new Path(args[1]).toUri());

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[2]));

        job.setMapperClass(SemiJoinMapper.class);
        job.setReducerClass(SemiJoinReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        return job.waitForCompletion(true) ? 0 : 1;
    }

}
