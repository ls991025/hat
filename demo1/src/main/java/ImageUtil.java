import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;

public class ImageUtil {
    private static File file=null;

    public static FileInputStream getImageByte(String infile) throws FileNotFoundException {
        FileInputStream imageByte=null;
        file=new File(infile);
        imageByte=new FileInputStream(file);
        return imageByte;
    }
}
