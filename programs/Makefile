FINITE_DIFFERENCE_TIME_DOMAIN_METHOD_DIRECTORY \
	:= finite_difference_time_domain_method/
FINITE_ELEMENT_METHOD_DIRECTORY := finite_element_method/


check: check_finite_difference_time_domain_method check_finite_element_method

check_finite_difference_time_domain_method:
	$(MAKE) check --directory $(FINITE_DIFFERENCE_TIME_DOMAIN_METHOD_DIRECTORY)

check_finite_element_method:
	$(MAKE) check --directory $(FINITE_ELEMENT_METHOD_DIRECTORY)


.PHONY: \
	check \
	check_finite_difference_time_domain_method \
	check_finite_element_method
