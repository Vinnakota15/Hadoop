import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import java.io.IOException;

public class SemiJoinReducer extends Reducer<Text, Text, Text, Text> {
    @Override
    protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        double totalAmount = 0.0;
        for (Text value : values) {
            totalAmount += Double.parseDouble(value.toString());
        }
        context.write(key, new Text(String.valueOf(totalAmount)));
    }
}
