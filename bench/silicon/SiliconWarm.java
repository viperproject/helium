// Silicon in one long-lived, warmed-up JVM: the driver behind `silicon_warm`
// (see bench/src/silicon_warm.rs, which starts it as
// `java <jvm args> -cp silicon.jar SiliconWarm.java <silicon args>`).
//
// Each file goes through Silicon's own command-line path
// (`SilFrontend.execute`, a fresh frontend, verifier and Z3 per file, exactly
// as `java -jar silicon.jar` runs it), so the output, the verdict and the time
// Silicon reports mean what they mean for a cold run; only the JVM is warm.
//
// Protocol, one command per stdin line, tab-separated; every answer ends with
// a line starting with `@@END `:
//
//   WARM <budget s> <per-file timeout s> <list file>
//       Verify the files listed (one path per line), in order and repeating
//       the list, until the budget is spent. Their output is discarded.
//       Answer: @@END WARM <runs> <seconds>
//   RUN <file.vpr>
//       Verify one file. Answer: Silicon's output, then
//       @@END RUN <ok|error> <seconds, as the driver measured them>
//   QUIT

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import scala.jdk.javaapi.CollectionConverters;
import viper.silicon.SiliconRunnerInstance;

public final class SiliconWarm {
    /** Where System.out (and so Scala's Console.out) currently goes. Scala
     *  captures System.out once, when Console is first used, so the stream is
     *  installed before any Silicon class loads and only its target changes. */
    static final class Switch extends OutputStream {
        volatile OutputStream target = OutputStream.nullOutputStream();

        @Override public void write(int b) throws IOException { target.write(b); }
        @Override public void write(byte[] b, int off, int len) throws IOException { target.write(b, off, len); }
        @Override public void flush() throws IOException { target.flush(); }
    }

    /** Starts every answer's last line; Silicon never prints it. */
    static final String END = "@@END ";
    static final Switch SWITCH = new Switch();
    static final PrintStream CAPTURED = new PrintStream(SWITCH, true, StandardCharsets.UTF_8);
    static List<String> baseArgs;

    public static void main(String[] args) throws IOException {
        PrintStream protocol = new PrintStream(new FileOutputStream(FileDescriptor.out), true, StandardCharsets.UTF_8);
        System.setOut(CAPTURED);
        System.setErr(CAPTURED);
        baseArgs = List.of(args);
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        protocol.println(END + "READY");
        String line;
        while ((line = in.readLine()) != null) {
            String[] f = line.split("\t");
            switch (f[0]) {
                case "WARM" -> {
                    double budget = Double.parseDouble(f[1]);
                    List<String> files = Files.readAllLines(Path.of(f[3]), StandardCharsets.UTF_8)
                        .stream().filter(s -> !s.isBlank()).toList();
                    List<String> extra = List.of("--timeout", f[2]);
                    long start = System.nanoTime();
                    int runs = 0;
                    while (!files.isEmpty() && seconds(start) < budget) {
                        verify(files.get(runs % files.size()), extra, OutputStream.nullOutputStream());
                        runs++;
                    }
                    protocol.println(END + "WARM " + runs + " " + seconds(start));
                }
                case "RUN" -> {
                    ByteArrayOutputStream out = new ByteArrayOutputStream();
                    // Start each file from a collected heap, as a cold JVM does;
                    // Silicon's own clock starts inside `execute`, after this.
                    System.gc();
                    long start = System.nanoTime();
                    boolean ok = verify(f[1], List.of(), out);
                    double secs = seconds(start);
                    protocol.print(out.toString(StandardCharsets.UTF_8));
                    protocol.println();
                    protocol.println(END + "RUN " + (ok ? "ok" : "error") + " " + secs);
                }
                case "QUIT" -> {
                    protocol.flush();
                    System.exit(0);
                }
                default -> protocol.println(END + "UNKNOWN " + f[0]);
            }
        }
        System.exit(0);
    }

    static double seconds(long start) {
        return (System.nanoTime() - start) / 1e9;
    }

    /** One file through Silicon's command-line path; false when it threw. */
    static boolean verify(String file, List<String> extra, OutputStream out) {
        SWITCH.target = out;
        SiliconRunnerInstance frontend = new SiliconRunnerInstance();
        List<String> args = new ArrayList<>(baseArgs);
        args.addAll(extra);
        args.add(file);
        try {
            frontend.execute(CollectionConverters.asScala(args).toSeq(), scala.Option.empty());
            return true;
        } catch (Throwable t) {
            t.printStackTrace(CAPTURED);
            return false;
        } finally {
            // Stops this file's Z3, as SiliconRunner does before it exits.
            try {
                if (frontend.verifier() != null) frontend.verifier().stop();
            } catch (Throwable t) {
                t.printStackTrace(CAPTURED);
            }
            CAPTURED.flush();
            SWITCH.target = OutputStream.nullOutputStream();
        }
    }
}
