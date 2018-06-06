from os.path import exists
import ubelt as ub


class CacheStamp(object):
    """
    Quickly determine if a computation that writes a file has been done.

    Writes a file that marks that a procedure has been done by writing a
    "stamp" file to disk. Removing the stamp file will force recomputation.
    However, removing or changing the result of the computation may not trigger
    recomputation unless specific handling is done or the expected "product"
    of the computation is a file and registered with the stamper.  If robust is
    False, we only check if the product exists, and we ignore its hash.

    Args:
        fname (str):
            name of the stamp file

        cfgstr (str):
            configuration associated with the stamped computation.  A common
            pattern is to call `ub.hash_data` on a dependency list.

        dpath (str):
            where to store the cached stamp file

        product (str or list, optional):
            Path or paths that we expect the computation to produce. If
            specified the hash of the paths are stored. It is faster, but lets
            robust if products are not specified.

        robust (bool):
            if True and product was specified, we use the product hash to
            check if it is expired, otherwise we assume the file has not
            been corrupoted or changed.

    Example:
        >>> import ubelt as ub
        >>> from os.path import join
        >>> # Stamp the computation of expensive-to-compute.txt
        >>> dpath = ub.ensure_app_cache_dir('netharn', 'test-cache-stemp')
        >>> product = join(dpath, 'expensive-to-compute.txt')
        >>> self = CacheStamp('somedata', 'someconfig', dpath, product)
        >>> if self.expired():
        >>>     ub.writeto(product, 'very expensive')
        >>>     self.renew()
        >>> assert not self.expired()
        >>> self.robust = False
        >>> # corrupting the output will not expire in non-robust mode
        >>> ub.writeto(product, 'corrupted')
        >>> assert not self.expired()
        >>> self.robust = True
        >>> # but it will expire if we are not in robust mode
        >>> assert self.expired()
        >>> # deleting the product will cause expiration in any mode
        >>> self.robust = False
        >>> ub.delete(product)
        >>> assert self.expired()
    """
    def __init__(self, fname, dpath, cfgstr=None, product=None, robust=True):
        self.cacher = ub.Cacher(fname, cfgstr=cfgstr, dpath=dpath)
        self.product = product
        self.robust = robust

    def _get_certificate(self, cfgstr=None):
        """
        Returns the stamp certificate if it exists
        """
        certificate = self.cacher.tryload(cfgstr=cfgstr)
        return certificate

    def _rectify_products(self, product=None):
        """ puts products in a normalied format """
        products = self.product if product is None else product
        if not isinstance(products, (list, tuple)):
            products = [products]
        return products

    def _product_file_hash(self, product=None):
        """
        Get the hash of the each product file
        """
        import xxhash  # much faster than builtin hashers
        products = self._rectify_products(product)
        product_file_hash = [ub.hash_file(p, hasher=xxhash.xxh64)
                             for p in products]
        return product_file_hash

    def expired(self, cfgstr=None, product=None):
        """
        Check to see if a previously existing stamp is still valid and if the
        expected result of that computation still exists.

        Args:
            cfgstr (str, optional): override the default cfgstr if specified
            product (str or list, optional): override the default product if
                specified
        """
        products = self._rectify_products(product)
        certificate = self._get_certificate(cfgstr=cfgstr)
        if certificate is None:
            # We dont even have a certificate, so we are expired
            is_expired = True
        elif products is None:
            # We dont have a product to check, so assume not expired
            # TODO: we could check the timestamp in the cerficiate
            is_expired = False
        elif not all(map(exists, products)):
            # We are expired if the expected product does not exist
            is_expired = True
        elif not self.robust:
            # Assume that the product hash is the same.
            is_expired = False
        else:
            # We are expired if the hash of the existing product data
            # does not match the expected hash in the certificate
            product_file_hash = self._product_file_hash(products)
            certificate_hash = certificate.get('product_file_hash', None)
            is_expired = product_file_hash != certificate_hash
        return is_expired

    def renew(self, cfgstr=None, product=None):
        """
        Recertify that the product has been recomputed by writing a new
        certificate to disk.
        """
        products = self._rectify_products(product)
        certificate = {
            'timestamp': ub.timestamp(),
            'product': products,
        }
        if products is not None:
            if not all(map(exists, products)):
                raise IOError(
                    'The stamped product must exist: {}'.format(products))
            certificate['product_file_hash'] = self._product_file_hash(products)
        self.cacher.save(certificate, cfgstr=cfgstr)


if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.util.util_cachestamp all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
