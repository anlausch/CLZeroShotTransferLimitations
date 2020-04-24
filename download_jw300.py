import itertools
import subprocess
import pexpect
import sys

print("Process started")
# install tool

#bash_command = "pip install opustools-pkg"
#print(bash_command)
#process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
#output, error = process.communicate()
#print(output)

#for comb in itertools.combinations(
#    ["ar", "eu", "cmn-Hans", "en", "fi", "he", "hi", "it", "ja", "ko", "ru", "sv", "tr"], 2):
# this is because the id listed for chinese is apparently wrong
for id in ["ar", "eu", "en", "fi", "he", "hi", "it", "ja", "ko", "ru", "sv", "tr"]:
    comb = ["tzh"]
    suff = "_" + comb[0] + "_" + comb[1]
    print("Working on: %s" % suff)
    try:
        bash_command = "opus_read -d JW300 -s " + comb[0] + " -t " + comb[1] + " -wm moses -w /work/anlausch/DebunkMLBERT/data/JW300/jw300" \
                       + suff + "." + comb[0] + " /work/anlausch/DebunkMLBERT/data/JW300/jw300" + suff + "." + comb[1]
        print(bash_command)
        child = pexpect.spawn(bash_command)
        #child.logfile_read = sys.stdout
        #print(child.read())
        child.expect('.*Continue.* ', timeout=10800)
        print(child.before)
        child.sendline('y')
        print(child.after)
        # timeout is three hours
        child.expect(pexpect.EOF, timeout=10800)
        print(child.before)
        child.close()

    except Exception as e:
        print(e)


