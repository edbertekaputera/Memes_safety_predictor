# Test
import  sys

def main():
	
	# Iteration loop to get new image filepath from sys.stdin:
	for line in sys.stdin:
		# IMPORTANT: Please ensure any trailing whitespace (eg: \n) is removed. This may impact some modules to open the filepath
		image_path = line.replace("\n", "")
		image_path = image_path.rstrip()
		if image_path == "":
			pass
		try:
			sys.stdout.write(f"{image_path}\n")
		except Exception as e:
			# Output to any raised/caught error/exceptions to stderr
			sys.stderr.write(str(e))


if __name__ == "__main__":
	main()